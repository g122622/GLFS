#include "core/gpu_index_adapter.h"

namespace glfs {

namespace backend_common {

inline bool is_valid_backend_type(const std::string& type) {
    return type == "rmi-cuda" || type == "g-index" || type == "cu-learned" ||
           type == "pgm-gpu" || type == "ligpu" || type == "cpu-rmi";
}

}  // namespace backend_common

}  // namespace glfs
