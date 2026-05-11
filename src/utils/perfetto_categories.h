#pragma once

#include "perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("glfs")
        .SetDescription("Top-level GLFS events"),
    perfetto::Category("glfs.fuse")
        .SetDescription("FUSE metadata and read-path callbacks"),
    perfetto::Category("glfs.lookup")
        .SetDescription("GPU/control-plane lookup path"),
    perfetto::Category("glfs.backing")
        .SetDescription("Backing-root fallback operations"),
    perfetto::Category("glfs.benchmark")
        .SetDescription("Benchmark driver and workload events"));
