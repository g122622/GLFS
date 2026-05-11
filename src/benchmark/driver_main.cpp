#include <chrono>
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
#include <unistd.h>

#include "benchmark/runner.h"
#include "config/config_manager.h"

namespace {

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
        if (is_fuse_mounted(path)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }
    throw std::runtime_error("timed out waiting for mount point: " + path.string());
}

void wait_for_process_exit(pid_t pid, std::uint32_t timeout_ms, std::uint32_t poll_interval_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
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

pid_t launch_mount_daemon(const std::filesystem::path& daemon_exe, const std::string& config_path) {
    if (!std::filesystem::exists(daemon_exe)) {
        throw std::runtime_error("mount daemon executable not found: " + daemon_exe.string());
    }

    const auto pid = ::fork();
    if (pid < 0) {
        throw std::runtime_error("failed to fork mount daemon process");
    }
    if (pid == 0) {
        ::execl(daemon_exe.c_str(), daemon_exe.c_str(), "--config", config_path.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    return pid;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--config") {
            std::cerr << "usage: glfs_benchmark --config FILE\n";
            return 1;
        }

        const std::filesystem::path config_path = argv[2];
        const auto cfg = glfs::load_config(config_path.string());
        const auto daemon_exe = std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path() / "gpufs");

        const auto daemon_pid = launch_mount_daemon(daemon_exe, config_path.string());
        try {
            wait_for_mount(cfg.fs.mount_point, daemon_pid, cfg.benchmark.mount_wait_timeout_ms, cfg.benchmark.mount_poll_interval_ms);

            const auto results = glfs::run_benchmarks(cfg);
            glfs::write_benchmark_report_csv(cfg.benchmark.report_csv_path, results);

            unmount_path(cfg.fs.mount_point);
        } catch (...) {
            try {
                unmount_path(cfg.fs.mount_point);
            } catch (...) {
            }
            wait_for_process_exit(daemon_pid, cfg.benchmark.daemon_stop_timeout_ms, cfg.benchmark.mount_poll_interval_ms);
            throw;
        }

        wait_for_process_exit(daemon_pid, cfg.benchmark.daemon_stop_timeout_ms, cfg.benchmark.mount_poll_interval_ms);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}