#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#include <signal.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>
#include <unistd.h>

#include "benchmark/runner.h"
#include "config/config_manager.h"
#include "utils/perfetto_integration.h"

namespace {

volatile std::sig_atomic_t g_interrupted = 0;

void signal_handler(int) {
    g_interrupted = 1;
    glfs::request_benchmark_stop();
    glfs::tracing_stop_session();
}

void install_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

void ensure_not_interrupted() {
    if (g_interrupted != 0 || glfs::benchmark_stop_requested()) {
        throw std::runtime_error("benchmark interrupted");
    }
}

bool is_fuse_mounted(const std::filesystem::path& path) {
    struct statfs sfs {};
    if (::statfs(path.c_str(), &sfs) != 0) {
        return false;
    }
#ifndef FUSE_SUPER_MAGIC
#define FUSE_SUPER_MAGIC 0x65735546
#endif
    return static_cast<unsigned long>(sfs.f_type) == static_cast<unsigned long>(FUSE_SUPER_MAGIC);
}

bool mount_point_ready(const std::filesystem::path& path) {
    if (!is_fuse_mounted(path)) {
        return false;
    }
    std::error_code ec;
    for (std::filesystem::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec)) {
        (void)it->path();
        break;
    }
    return !ec;
}

void unmount_path(const std::filesystem::path& path) {
    const std::string quoted = "\"" + path.string() + "\"";
    std::string cmd = "fusermount3 -u " + quoted;
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        cmd = "fusermount -u " + quoted;
        rc = std::system(cmd.c_str());
    }
    if (rc != 0) {
        throw std::runtime_error("failed to unmount: " + path.string());
    }
}

void wait_for_mount(const std::filesystem::path& path,
                    pid_t daemon_pid,
                    std::uint32_t timeout_ms,
                    std::uint32_t poll_interval_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        ensure_not_interrupted();
        int status = 0;
        const auto waited = ::waitpid(daemon_pid, &status, WNOHANG);
        if (waited == daemon_pid) {
            if (WIFEXITED(status)) {
                throw std::runtime_error("mount daemon exited early with status " + std::to_string(WEXITSTATUS(status)));
            }
            if (WIFSIGNALED(status)) {
                throw std::runtime_error("mount daemon terminated early by signal " + std::to_string(WTERMSIG(status)));
            }
            throw std::runtime_error("mount daemon exited early");
        }
        if (mount_point_ready(path)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }
    throw std::runtime_error("timed out waiting for mount point: " + path.string());
}

void wait_for_process_exit(pid_t pid, std::uint32_t timeout_ms, std::uint32_t poll_interval_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (g_interrupted != 0) {
            break;
        }
        int status = 0;
        const auto waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }
    if (::kill(pid, SIGKILL) == 0) {
        (void)::waitpid(pid, nullptr, 0);
    }
}

pid_t launch_mount_daemon(const std::filesystem::path& daemon_exe, const std::string& config_path, int ready_fd) {
    if (!std::filesystem::exists(daemon_exe)) {
        throw std::runtime_error("mount daemon executable not found: " + daemon_exe.string());
    }

    const auto pid = ::fork();
    if (pid < 0) {
        throw std::runtime_error("failed to fork mount daemon process");
    }
    if (pid == 0) {
        (void)::setpgid(0, 0);
        const std::string ready_fd_arg = std::to_string(ready_fd);
        ::execl(daemon_exe.c_str(), daemon_exe.c_str(), "--config", config_path.c_str(), "--ready-fd", ready_fd_arg.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    (void)::setpgid(pid, pid);
    return pid;
}

void terminate_daemon(pid_t pid, std::uint32_t timeout_ms, std::uint32_t poll_interval_ms) {
    if (pid <= 0) {
        return;
    }
    (void)::kill(-pid, SIGTERM);
    wait_for_process_exit(pid, timeout_ms + 2000, poll_interval_ms);
    if (::kill(pid, 0) == 0) {
        (void)::kill(-pid, SIGKILL);
        (void)::waitpid(pid, nullptr, 0);
    }
}

void wait_for_mount_ready_fd(int fd, pid_t daemon_pid, std::uint32_t timeout_ms) {
    struct pollfd pfd {};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int rc = ::poll(&pfd, 1, static_cast<int>(timeout_ms));
    if (rc <= 0) {
        int status = 0;
        const auto waited = ::waitpid(daemon_pid, &status, WNOHANG);
        if (waited == daemon_pid) {
            if (WIFEXITED(status)) {
                throw std::runtime_error("mount daemon exited early with status " + std::to_string(WEXITSTATUS(status)));
            }
            if (WIFSIGNALED(status)) {
                throw std::runtime_error("mount daemon terminated early by signal " + std::to_string(WTERMSIG(status)));
            }
        }
        throw std::runtime_error("timed out waiting for mount daemon ready signal");
    }

    char ready = 0;
    if (::read(fd, &ready, 1) != 1 || ready != '1') {
        throw std::runtime_error("mount daemon failed to signal mount readiness");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        install_signal_handlers();
        glfs::reset_benchmark_stop();
        glfs::tracing_init();

        if (argc != 3 || std::string(argv[1]) != "--config") {
            std::cerr << "usage: glfs_benchmark --config FILE\n";
            return 1;
        }

        const std::filesystem::path config_path = argv[2];
        const auto cfg = glfs::load_config(config_path.string());
        const auto daemon_exe = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path() / "gpufs");
        int ready_pipe[2] = {-1, -1};
        if (::pipe(ready_pipe) != 0) {
            throw std::runtime_error("failed to create mount daemon ready pipe");
        }

        const auto daemon_pid = launch_mount_daemon(daemon_exe, config_path.string(), ready_pipe[1]);
        (void)::close(ready_pipe[1]);
        try {
            wait_for_mount_ready_fd(ready_pipe[0], daemon_pid, cfg.benchmark.mount_wait_timeout_ms);
            (void)::close(ready_pipe[0]);
            wait_for_mount(cfg.fs.mount_point, daemon_pid, cfg.benchmark.mount_wait_timeout_ms, cfg.benchmark.mount_poll_interval_ms);

            ensure_not_interrupted();

            const auto results = glfs::run_benchmarks(cfg);
            glfs::write_benchmark_report_csv(cfg.benchmark.report_csv_path, results);

            unmount_path(cfg.fs.mount_point);
        } catch (...) {
            if (ready_pipe[0] >= 0) {
                (void)::close(ready_pipe[0]);
            }
            try {
                unmount_path(cfg.fs.mount_point);
            } catch (...) {
            }
            terminate_daemon(daemon_pid, cfg.benchmark.daemon_stop_timeout_ms, cfg.benchmark.mount_poll_interval_ms);
            throw;
        }

        terminate_daemon(daemon_pid, cfg.benchmark.daemon_stop_timeout_ms, cfg.benchmark.mount_poll_interval_ms);

        glfs::tracing_shutdown();
        return 0;
    } catch (const std::exception& ex) {
        glfs::tracing_shutdown();
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}
