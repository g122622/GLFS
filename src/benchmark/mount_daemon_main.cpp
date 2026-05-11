#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"
#include "fuse/gpufs_ops.h"

int main(int argc, char** argv) {
    try {
        std::string config_path;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];
            if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
                continue;
            }
            if (arg == "--help" || arg == "-h") {
                std::cout << "usage: glfs_mount_daemon --config FILE\n";
                return 0;
            }
            throw std::runtime_error(std::string("unsupported command-line argument: ") + std::string(arg));
        }

        if (config_path.empty()) {
            throw std::runtime_error("missing required --config FILE");
        }

        const auto cfg = glfs::load_config(config_path);
        if (cfg.fs.mount_point.empty()) {
            throw std::runtime_error("mount_point is empty");
        }

        std::error_code ec;
        if (std::filesystem::exists(cfg.fs.mount_point, ec)) {
            if (!std::filesystem::is_directory(cfg.fs.mount_point, ec)) {
                throw std::runtime_error("mount_point exists but is not a directory: " + cfg.fs.mount_point);
            }
        } else {
            if (!std::filesystem::create_directories(cfg.fs.mount_point, ec) && ec) {
                throw std::runtime_error("failed to create mount_point: " + cfg.fs.mount_point + ": " + ec.message());
            }
        }

        std::unique_ptr<glfs::IGPUControlPlane, void (*)(glfs::IGPUControlPlane*)> control_plane(
            glfs::create_control_plane(cfg.index.type), glfs::destroy_control_plane);
        glfs::GPULearnedFS fs{};
        glfs::gpufs_init(fs, control_plane.get(), cfg);
        glfs::set_active_fs(&fs);

        std::vector<std::string> storage;
        storage.reserve(cfg.fs.fuse_opts.size() + 2);
        storage.push_back(argv[0]);
        storage.push_back("-f");
        for (const auto& opt : cfg.fs.fuse_opts) {
            storage.push_back(opt);
        }
        storage.push_back(cfg.fs.mount_point);

        std::vector<char*> fuse_args;
        fuse_args.reserve(storage.size());
        for (auto& s : storage) {
            fuse_args.push_back(s.data());
        }

        std::cout << "mounting...\n";
        auto ops = glfs::build_fuse_operations();
        int ret = fuse_main(static_cast<int>(fuse_args.size()), fuse_args.data(), &ops, nullptr);
        return ret == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}