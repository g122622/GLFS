#include <cassert>
#include <iostream>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"
#include "core/path_encoder.h"
#include "utils/json_parser.h"

int main() {
    using namespace glfs;

    const auto j = parse_json(R"({"a":1,"b":[true,"x"]})");
    assert(j.contains("a"));
    assert(j.at("a").as_number() == 1.0);
    assert(j.at("b").at(0).as_bool());
    assert(j.at("b").at(1).as_string() == "x");

    auto normalized = normalize_path("/home/user/data/./train/../train/img");
    assert(normalized == "/home/user/data/train/img");

    PathConfig pc;
    pc.mount_point = "/home/user/data";
    pc.max_depth = 32;
    pc.bits_per_level = 8;
    auto key = encode_path("/home/user/data/train/img/cat.jpg", pc);
    assert(key.value != EncodedKey::INVALID_KEY);

    TrainingConfig tcfg;
    tcfg.index_type = "g-index";
    tcfg.sample_ratio = 1.0f;
    tcfg.max_epochs = 1;
    tcfg.max_vram_mb = 256;
    tcfg.segment_base_width = 256;
    tcfg.segment_min_width = 32;
    tcfg.segment_max_width = 4096;
    tcfg.segment_epoch_cap = 8;
    tcfg.lookup_window = 32;
    tcfg.cuda_block_size = 256;
    tcfg.latency_history_limit = 1024;
    tcfg.vram_overhead_bytes = 1024;

    auto* idx = create_index("g-index");
    idx->train({1, 2, 3}, {10, 20, 30}, tcfg);
    auto out = idx->batch_lookup({2, 4});
    assert(out[0] == 20);
    assert(out[1] == INVALID_INODE);
    destroy_index(idx);

    std::cout << "all tests passed\n";
    return 0;
}
