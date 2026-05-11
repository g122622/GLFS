#include "utils/perfetto_integration.h"

#include <fstream>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace glfs {

namespace {

struct TraceRuntime {
    bool initialized = false;
    bool session_active = false;
    std::string session_name = "glfs";
    std::string output_path = "trace_glfs.perfetto-trace";
    std::unique_ptr<::perfetto::TracingSession> session;
    int output_fd = -1;
};

std::mutex& trace_mutex() {
    static std::mutex mutex;
    return mutex;
}

TraceRuntime& trace_runtime() {
    static TraceRuntime runtime;
    return runtime;
}

::perfetto::TraceConfig make_trace_config(std::uint32_t buffer_size_kb, bool write_into_file) {
    ::perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(buffer_size_kb);
    cfg.set_flush_timeout_ms(5000);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    ::perfetto::protos::gen::TrackEventConfig te_cfg;
    te_cfg.add_disabled_categories("*");
    te_cfg.add_enabled_categories("glfs");
    te_cfg.add_enabled_categories("glfs.fuse");
    te_cfg.add_enabled_categories("glfs.lookup");
    te_cfg.add_enabled_categories("glfs.backing");
    te_cfg.add_enabled_categories("glfs.benchmark");
    ds_cfg->set_track_event_config_raw(te_cfg.SerializeAsString());
    if (write_into_file) {
        cfg.set_write_into_file(true);
        cfg.set_file_write_period_ms(1000);
    }
    return cfg;
}

void write_trace_to_file_locked() {
    auto& runtime = trace_runtime();
    if (!runtime.session) {
        return;
    }
    if (runtime.output_fd >= 0) {
        return;
    }
    const std::vector<char> trace_data = runtime.session->ReadTraceBlocking();
    if (runtime.output_path.empty()) {
        return;
    }
    std::ofstream out(runtime.output_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    out.write(trace_data.data(), static_cast<std::streamsize>(trace_data.size()));
}

}  // namespace

void tracing_init() {
    std::lock_guard<std::mutex> lock(trace_mutex());
    auto& runtime = trace_runtime();
    if (runtime.initialized) {
        return;
    }

    ::perfetto::TracingInitArgs args;
    args.backends |= ::perfetto::kInProcessBackend;
    args.backends |= ::perfetto::kSystemBackend;
    args.enable_system_consumer = false;
    ::perfetto::Tracing::Initialize(args);
    ::perfetto::TrackEvent::Register();
    runtime.initialized = true;
}

void tracing_shutdown() {
    tracing_stop_session();
}

void tracing_start_session(const TraceSessionOptions& options) {
    std::lock_guard<std::mutex> lock(trace_mutex());
    auto& runtime = trace_runtime();
    if (!runtime.initialized) {
        ::perfetto::TracingInitArgs args;
        args.backends |= ::perfetto::kInProcessBackend;
        args.backends |= ::perfetto::kSystemBackend;
        args.enable_system_consumer = false;
        ::perfetto::Tracing::Initialize(args);
        ::perfetto::TrackEvent::Register();
        runtime.initialized = true;
    }

    if (runtime.session_active) {
        runtime.session->StopBlocking();
        write_trace_to_file_locked();
        runtime.session.reset();
        if (runtime.output_fd >= 0) {
            ::close(runtime.output_fd);
            runtime.output_fd = -1;
        }
        runtime.session_active = false;
    }

    runtime.session_name = options.session_name.empty() ? "glfs" : options.session_name;
    runtime.output_path = options.output_path.empty() ? "trace_glfs.perfetto-trace" : options.output_path;
    runtime.session = ::perfetto::Tracing::NewTrace(::perfetto::kInProcessBackend);
    if (options.write_into_file) {
        runtime.output_fd = ::open(runtime.output_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (runtime.output_fd < 0) {
            throw std::runtime_error("failed to open trace output file: " + runtime.output_path);
        }
        runtime.session->Setup(make_trace_config(options.buffer_size_kb, true), runtime.output_fd);
    } else {
        runtime.session->Setup(make_trace_config(options.buffer_size_kb, false));
    }
    runtime.session->StartBlocking();
    runtime.session_active = true;
}

void tracing_stop_session() {
    std::lock_guard<std::mutex> lock(trace_mutex());
    auto& runtime = trace_runtime();
    if (!runtime.session_active || !runtime.session) {
        return;
    }
    ::perfetto::TrackEvent::Flush();
    runtime.session->FlushBlocking(5000);
    runtime.session->StopBlocking();
    write_trace_to_file_locked();
    runtime.session.reset();
    if (runtime.output_fd >= 0) {
        ::close(runtime.output_fd);
        runtime.output_fd = -1;
    }
    runtime.session_active = false;
}

void tracing_flush() {
    std::lock_guard<std::mutex> lock(trace_mutex());
    auto& runtime = trace_runtime();
    if (!runtime.session_active || !runtime.session) {
        return;
    }
    runtime.session->FlushBlocking(5000);
}

bool tracing_session_active() {
    std::lock_guard<std::mutex> lock(trace_mutex());
    return trace_runtime().session_active;
}

std::string tracing_output_path() {
    std::lock_guard<std::mutex> lock(trace_mutex());
    return trace_runtime().output_path;
}

void perfetto_track_event(const char* name, std::int64_t, std::int64_t) {
    if (name) {
        TRACE_EVENT_INSTANT("glfs", perfetto::DynamicString{name});
    } else {
        TRACE_EVENT_INSTANT("glfs", "event");
    }
}

void perfetto_flush(const std::string& filepath) {
    {
        std::lock_guard<std::mutex> lock(trace_mutex());
        auto& runtime = trace_runtime();
        if (!filepath.empty()) {
            runtime.output_path = filepath;
        }
    }
    tracing_stop_session();
}

}  // namespace glfs
