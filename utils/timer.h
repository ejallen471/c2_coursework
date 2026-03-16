/**
 * @file timer.h
 * @brief Minimal wall-clock timing utility used by the benchmark drivers.
 */

#ifndef TIMER_H
#define TIMER_H

/**
 * @brief Returns a monotonic wall-clock timestamp in seconds.
 * @return Seconds since the steady clock epoch.
 */
double wall_time_seconds();

#endif
