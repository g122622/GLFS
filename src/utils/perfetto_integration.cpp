#include "utils/perfetto_integration.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "utils/timer.h"

namespace glfs {

namespace {

struct TraceEvent {
    std::string name;
    std::int64_t start_ns;
    std::int64_t dur_ns;
};

std::mutex& trace_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<TraceEvent>& trace_events() {
    static std::vector<TraceEvent> events;
    return events;
}

std::string& trace_session_name() {
    static std::string name = "glfs";
    return name;
}

std::string escape_json(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

}  // namespace

void tracing_init(const std::string& session_name) {
    std::lock_guard<std::mutex> lock(trace_mutex());
    trace_session_name() = session_name;
    trace_events().clear();
}

void perfetto_track_event(const char* name, std::int64_t start_ns, std::int64_t dur_ns) {
    std::lock_guard<std::mutex> lock(trace_mutex());
    trace_events().push_back(TraceEvent{name ? name : "event", start_ns, dur_ns});
}

void perfetto_flush(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(trace_mutex());
    std::ofstream out(filepath, std::ios::trunc);
    out << "{\n";
    out << "  \"displayTimeUnit\": \"ns\",\n";
    out << "  \"session\": \"" << escape_json(trace_session_name()) << "\",\n";
    out << "  \"traceEvents\": [\n";
    for (std::size_t i = 0; i < trace_events().size(); ++i) {
        const auto& ev = trace_events()[i];
        out << "    {\"name\":\"" << escape_json(ev.name) << "\","
            << "\"ph\":\"X\","
            << "\"ts\":" << ev.start_ns << ","
            << "\"dur\":" << ev.dur_ns << ","
            << "\"pid\":1,\"tid\":1}";
        if (i + 1 != trace_events().size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace glfs
