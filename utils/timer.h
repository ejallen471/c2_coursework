/**
 * @file timer.h
 * @brief Helper for measuring elapsed time.
 */

#ifndef TIMER_H
#define TIMER_H

/**
 * @brief Get the current time from a steady (monotonic) clock.
 *
 * This is used for timing code.
 * but differences between two calls give elapsed time in seconds.
 *
 * @return Current time in seconds.
 */
double wall_time_seconds();

#endif