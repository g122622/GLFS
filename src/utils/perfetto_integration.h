#pragma once

#include <cstdint>
#include <string>

#include "utils/perfetto_categories.h"

namespace glfs {

struct TraceSessionOptions {
    std::string session_name = "glfs";
    std::string output_path = "trace_glfs.perfetto-trace";
    std::uint32_t buffer_size_kb = 16384;
    bool write_into_file = false;
};

void tracing_init();
void tracing_shutdown();

void tracing_start_session(const TraceSessionOptions& options = {});
void tracing_stop_session();
void tracing_flush();
bool tracing_session_active();
std::string tracing_output_path();

void perfetto_track_event(const char* name, std::int64_t start_ns, std::int64_t dur_ns);
void perfetto_flush(const std::string& filepath = "");

}  // namespace glfs
