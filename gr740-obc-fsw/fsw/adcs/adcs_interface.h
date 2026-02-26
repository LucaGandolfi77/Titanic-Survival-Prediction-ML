/**
 * @file adcs_interface.h
 * @brief Attitude Determination and Control Subsystem Interface API.
 *
 * Communicates with external ADCS unit via CAN bus.
 * Provides attitude quaternion, angular rates, and mode control.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef ADCS_INTERFACE_H
#define ADCS_INTERFACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define ADCS_OK            0
#define ADCS_ERR_PARAM    -1
#define ADCS_ERR_COMM     -2
#define ADCS_ERR_TIMEOUT  -3

/* ── ADCS mode ─────────────────────────────────────────────────────────── */
typedef enum {
    ADCS_MODE_OFF        = 0,
    ADCS_MODE_DETUMBLE   = 1,
    ADCS_MODE_SUN_POINT  = 2,
    ADCS_MODE_NADIR      = 3,
    ADCS_MODE_TARGET     = 4
} adcs_mode_t;

/* ── ADCS telemetry ────────────────────────────────────────────────────── */
typedef struct {
    /* Quaternion (q0 scalar, q1-q3 vector), scaled ×10000 */
    int16_t  q0;
    int16_t  q1;
    int16_t  q2;
    int16_t  q3;
    /* Angular rates (deg/s × 100) */
    int16_t  omega_x;
    int16_t  omega_y;
    int16_t  omega_z;
    /* Status */
    adcs_mode_t mode;
    uint8_t     pointing_ok;     /**< 1 = within pointing budget  */
    uint8_t     tumbling;        /**< 1 = tumbling detected        */
    int16_t     sun_angle_01deg; /**< Sun angle (0.1°)             */
} adcs_telemetry_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the ADCS interface.
 * @return ADCS_OK on success.
 */
int32_t adcs_init(void);

/**
 * @brief Request current ADCS telemetry via CAN.
 *
 * @param[out] tlm  Telemetry output.
 * @return ADCS_OK on success.
 */
int32_t adcs_get_telemetry(adcs_telemetry_t *tlm);

/**
 * @brief Command ADCS mode.
 *
 * @param[in] mode  Target ADCS mode.
 * @return ADCS_OK on success.
 */
int32_t adcs_set_mode(adcs_mode_t mode);

/**
 * @brief Get current ADCS mode.
 * @return Current mode.
 */
adcs_mode_t adcs_get_mode(void);

/**
 * @brief Check if spacecraft is tumbling.
 * @return 1 if tumbling, 0 if stable.
 */
int32_t adcs_is_tumbling(void);

/**
 * @brief Periodic ADCS tick — poll telemetry, check pointing.
 *
 * @return ADCS_OK on success.
 */
int32_t adcs_tick(void);

#ifdef __cplusplus
}
#endif

#endif /* ADCS_INTERFACE_H */
