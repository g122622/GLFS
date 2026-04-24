#pragma once

#include <cstdint>
#include <string>

namespace glfs {

void tracing_init(const std::string& session_name = "glfs");
void perfetto_track_event(const char* name, std::int64_t start_ns, std::int64_t dur_ns);
void perfetto_flush(const std::string& filepath = "trace_glfs.perfetto.json");

}  // namespace glfs
