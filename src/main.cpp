#include <memory>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"
#include "core/path_encoder.h"
#include "fuse/gpufs_ops.h"
#include "utils/perfetto_integration.h"

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
                std::cout << "usage: gpufs --config FILE\n";
                return 0;
            }
            throw std::runtime_error(std::string("unsupported command-line argument: ") + std::string(arg));
        }

        if (config_path.empty()) {
            throw std::runtime_error("missing required --config FILE");
        }

        auto cfg = glfs::load_config(config_path);

        if (cfg.fs.mount_point.empty()) {
            throw std::runtime_error("mount_point is empty");
        }

        glfs::tracing_init("gpufs");

        std::unique_ptr<glfs::IGPUControlPlane, void (*)(glfs::IGPUControlPlane*)> control_plane(
            glfs::create_control_plane(cfg.index.type), glfs::destroy_control_plane);
        glfs::GPULearnedFS fs{};
        glfs::gpufs_init(fs, control_plane, cfg);
        glfs::set_active_fs(&fs);

        std::cout << "GPULearnedFS mount entry ready\n";
        std::cout << "mount_point=" << cfg.fs.mount_point << '\n';
        std::cout << "index_type=" << cfg.index.type << '\n';
        std::cout << "backing_root=" << cfg.fs.backing_root << '\n';
        std::cout << "strict_mode=" << (cfg.fs.strict_mode ? "true" : "false") << '\n';

        auto ops = glfs::build_fuse_operations();
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
        int ret = fuse_main(static_cast<int>(fuse_args.size()), fuse_args.data(), &ops, nullptr);
        return ret == 0 ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << '\n';
        return 1;
    }
}
