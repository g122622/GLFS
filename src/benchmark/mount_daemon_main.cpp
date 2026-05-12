#include <csignal>
#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

#include <sys/statfs.h>
#include <unistd.h>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"
#include "fuse/gpufs_ops.h"
#include "utils/perfetto_integration.h"

namespace {

glfs::TrainingConfig make_training_config(const glfs::FSConfig& cfg) {
    glfs::TrainingConfig tcfg;
    tcfg.index_type = cfg.index.type;
    tcfg.sample_ratio = cfg.index.training.sample_ratio;
    tcfg.max_epochs = cfg.index.training.max_epochs;
    tcfg.max_vram_mb = static_cast<std::size_t>(cfg.index.resource.max_vram_bytes / (1024ULL * 1024ULL));
    tcfg.segment_base_width = cfg.index.backend.segment_base_width;
    tcfg.segment_min_width = cfg.index.backend.segment_min_width;
    tcfg.segment_max_width = cfg.index.backend.segment_max_width;
    tcfg.segment_epoch_cap = cfg.index.backend.segment_epoch_cap;
    tcfg.lookup_window = cfg.index.backend.lookup_window;
    tcfg.cuda_block_size = cfg.index.backend.cuda_block_size;
    tcfg.latency_history_limit = cfg.index.backend.latency_history_limit;
    tcfg.vram_overhead_bytes = cfg.index.backend.vram_overhead_bytes;
    return tcfg;
}

volatile std::sig_atomic_t g_shutdown_requested = 0;
struct fuse* g_fuse_handle = nullptr;

void signal_handler(int) {
    g_shutdown_requested = 1;
    glfs::tracing_stop_session();
    if (g_fuse_handle) {
        fuse_exit(g_fuse_handle);
    }
}

void install_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

void shutdown_tracing_at_exit() {
    glfs::tracing_shutdown();
}

void release_resources(glfs::IGPUControlPlane* control_plane, glfs::GPULearnedFS* fs) {
    if (fs) {
        glfs::set_active_fs(nullptr);
    }
    if (control_plane) {
        glfs::destroy_control_plane(control_plane);
    }
}

void force_reset_mount_point(const std::string& mount_point) {
    const std::string quoted = "\"" + mount_point + "\"";
    (void)std::system(("fusermount3 -uz " + quoted).c_str());
    (void)std::system(("fusermount -uz " + quoted).c_str());

    std::error_code ec;
    const std::filesystem::path path(mount_point);
    std::filesystem::remove_all(path, ec);
    ec.clear();
    if (!std::filesystem::create_directories(path, ec) && ec) {
        throw std::runtime_error("failed to recreate mount_point: " + mount_point + ": " + ec.message());
    }
    if (::chmod(mount_point.c_str(), 0755) != 0) {
        throw std::runtime_error("failed to chmod mount_point: " + mount_point + ": " + std::string(std::strerror(errno)));
    }
}

void prepare_mount_point(const std::string& mount_point) {
    if (mount_point.empty()) {
        throw std::runtime_error("mount_point is empty");
    }

    std::error_code ec;
    const std::filesystem::path path(mount_point);
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        force_reset_mount_point(mount_point);
        return;
    }

    if (!exists) {
        force_reset_mount_point(mount_point);
        return;
    }

    if (!std::filesystem::is_directory(path, ec)) {
        force_reset_mount_point(mount_point);
        return;
    }

    if (::access(mount_point.c_str(), R_OK | W_OK | X_OK) != 0) {
        force_reset_mount_point(mount_point);
        return;
    }

    if (::chmod(mount_point.c_str(), 0755) != 0) {
        force_reset_mount_point(mount_point);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        install_signal_handlers();
        ::atexit(shutdown_tracing_at_exit);
        std::string config_path;
        std::string ready_fd_value;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
                continue;
            }
            if (arg == "--ready-fd" && i + 1 < argc) {
                ready_fd_value = argv[++i];
                continue;
            }
            if (arg == "--help" || arg == "-h") {
                std::cout << "usage: glfs_mount_daemon --config FILE [--ready-fd N]\n";
                return 0;
            }
            throw std::runtime_error(std::string("unsupported command-line argument: ") + std::string(arg));
        }

        if (config_path.empty()) {
            throw std::runtime_error("missing required --config FILE");
        }
        int ready_fd = -1;
        if (!ready_fd_value.empty()) {
            ready_fd = std::stoi(ready_fd_value);
        }

        const auto cfg = glfs::load_config(config_path);
        glfs::tracing_init();
        glfs::TraceSessionOptions trace_options;
        trace_options.session_name = "glfs-mount-daemon";
        trace_options.output_path = "trace_glfs_daemon.perfetto-trace";
        trace_options.write_into_file = true;
        glfs::tracing_start_session(trace_options);
        prepare_mount_point(cfg.fs.mount_point);

        const auto training_cfg = make_training_config(cfg);
        std::unique_ptr<glfs::IGPUControlPlane, void (*)(glfs::IGPUControlPlane*)> control_plane(
            glfs::create_control_plane(cfg.index.type,
                                       training_cfg,
                                       cfg.index.inference.batch_size,
                                       cfg.index.inference.batch_timeout_us),
            glfs::destroy_control_plane);
        glfs::GPULearnedFS fs{};
        glfs::gpufs_init(fs, control_plane.get(), cfg);
        glfs::set_active_fs(&fs);

        std::vector<std::string> storage;
        storage.reserve(cfg.fs.fuse_opts.size() + 1);
        storage.push_back(argv[0]);
        for (const auto& opt : cfg.fs.fuse_opts) {
            storage.push_back(opt);
        }

        std::vector<char*> fuse_args;
        fuse_args.reserve(storage.size());
        for (auto& s : storage) {
            fuse_args.push_back(s.data());
        }

        std::cout << "mounting...\n";
        auto ops = glfs::build_fuse_operations();
        int ret = 0;
        try {
            struct fuse_args fargs = FUSE_ARGS_INIT(static_cast<int>(fuse_args.size()), fuse_args.data());
            g_fuse_handle = fuse_new(&fargs, &ops, sizeof(ops), nullptr);
            if (!g_fuse_handle) {
                fuse_opt_free_args(&fargs);
                throw std::runtime_error("failed to create FUSE instance");
            }
            if (fuse_mount(g_fuse_handle, cfg.fs.mount_point.c_str()) != 0) {
                fuse_opt_free_args(&fargs);
                fuse_destroy(g_fuse_handle);
                g_fuse_handle = nullptr;
                throw std::runtime_error("failed to mount FUSE filesystem");
            }
            std::cout << "mounted: " << cfg.fs.mount_point << '\n';
            std::cout.flush();
            if (ready_fd >= 0) {
                const char ready_byte = '1';
                (void)::write(ready_fd, &ready_byte, 1);
                (void)::close(ready_fd);
                ready_fd = -1;
            }
            ret = fuse_loop(g_fuse_handle);
            fuse_opt_free_args(&fargs);
        } catch (...) {
            if (ready_fd >= 0) {
                (void)::close(ready_fd);
            }
            if (g_fuse_handle) {
                fuse_unmount(g_fuse_handle);
                fuse_destroy(g_fuse_handle);
                g_fuse_handle = nullptr;
            }
            release_resources(control_plane.release(), &fs);
            throw;
        }

        if (g_fuse_handle) {
            fuse_unmount(g_fuse_handle);
            fuse_destroy(g_fuse_handle);
            g_fuse_handle = nullptr;
        }

        release_resources(control_plane.release(), &fs);
        glfs::tracing_shutdown();
        return ret == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        glfs::tracing_shutdown();
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}
