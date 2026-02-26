/**
 * @file health_monitor.h
 * @brief Health Monitor — task and subsystem health tracking.
 *
 * Aggregates watchdog heartbeat status with subsystem health
 * indicators and exposes a consolidated system health vector
 * consumed by the FDIR module.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef HEALTH_MONITOR_H
#define HEALTH_MONITOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define HMON_OK            0
#define HMON_ERR_PARAM    -1

/* ── Configuration ─────────────────────────────────────────────────────── */
#define HMON_MAX_ITEMS     32U

/* ── Health status values ──────────────────────────────────────────────── */
typedef enum {
    HEALTH_UNKNOWN   = 0,
    HEALTH_NOMINAL   = 1,
    HEALTH_DEGRADED  = 2,
    HEALTH_FAULTY    = 3,
    HEALTH_OFFLINE   = 4
} health_status_t;

/* ── Subsystem identifiers ─────────────────────────────────────────────── */
typedef enum {
    HMON_SUBSYS_OBC    =  0,
    HMON_SUBSYS_EPS    =  1,
    HMON_SUBSYS_ADCS   =  2,
    HMON_SUBSYS_COMMS  =  3,
    HMON_SUBSYS_PAYLOAD=  4,
    HMON_SUBSYS_THERMAL=  5,
    HMON_SUBSYS_SPW    =  6,
    HMON_SUBSYS_CAN    =  7,
    HMON_SUBSYS_MEM    =  8,
    HMON_SUBSYS_COUNT  =  9
} hmon_subsys_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the health monitor.
 * @return HMON_OK on success.
 */
int32_t hmon_init(void);

/**
 * @brief Update health status of a subsystem.
 *
 * @param[in] subsys  Subsystem identifier.
 * @param[in] status  Health status value.
 * @return HMON_OK on success.
 */
int32_t hmon_set_status(hmon_subsys_t subsys, health_status_t status);

/**
 * @brief Get health status of a subsystem.
 *
 * @param[in]  subsys  Subsystem identifier.
 * @param[out] status  Pointer to receive status.
 * @return HMON_OK on success.
 */
int32_t hmon_get_status(hmon_subsys_t subsys, health_status_t *status);

/**
 * @brief Get overall system health.
 *
 * Returns the worst-case status across all monitored subsystems.
 *
 * @return Overall health status.
 */
health_status_t hmon_get_overall(void);

/**
 * @brief Periodic health check tick.
 *
 * Queries the watchdog module and updates health statuses.
 * Call every minor frame.
 *
 * @return Number of faulty subsystems detected.
 */
int32_t hmon_tick(void);

/**
 * @brief Get the health status vector as a bitmask.
 *
 * Bit layout: bits [1:0] per subsystem, 2 bits each.
 * bit 0=1 means nominal, 11 means faulty.
 *
 * @return 32-bit health vector.
 */
uint32_t hmon_get_vector(void);

#ifdef __cplusplus
}
#endif

#endif /* HEALTH_MONITOR_H */
