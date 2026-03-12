#include "timer.h"
#include <chrono>

double wall_time_seconds()
{
    // Use a monotonic clock so measured intervals are not affected by wall-clock adjustments.
    using clock = std::chrono::steady_clock;

    // Express the current time point as a duration since the clock epoch.
    const auto now = clock::now().time_since_epoch();

    // Convert that duration to seconds as a double for simple elapsed-time subtraction.
    return std::chrono::duration<double>(now).count();
}
