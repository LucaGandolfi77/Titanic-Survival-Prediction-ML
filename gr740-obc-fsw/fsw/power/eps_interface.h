/**
 * @file eps_interface.h
 * @brief Electrical Power Subsystem (EPS) Interface API.
 *
 * Communicates with the EPS unit via CAN bus (CANopen protocol).
 * Provides battery voltage/current, solar panel status, and
 * power switch control.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef EPS_INTERFACE_H
#define EPS_INTERFACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define EPS_OK            0
#define EPS_ERR_PARAM    -1
#define EPS_ERR_COMM     -2
#define EPS_ERR_TIMEOUT  -3

/* ── Configuration ─────────────────────────────────────────────────────── */
#define EPS_MAX_CHANNELS   8U    /**< Power switch channels             */

/* ── EPS telemetry structure ───────────────────────────────────────────── */
typedef struct {
    uint16_t battery_voltage_mv;   /**< Battery voltage (mV)            */
    int16_t  battery_current_ma;   /**< Battery current (mA, +charge)   */
    uint8_t  battery_soc_pct;      /**< State of charge (%)             */
    int16_t  battery_temp_01c;     /**< Battery temperature (0.1°C)     */
    uint16_t solar_voltage_mv;     /**< Solar array voltage (mV)        */
    int16_t  solar_current_ma;     /**< Solar array current (mA)        */
    uint16_t bus_voltage_mv;       /**< Main bus voltage (mV)           */
    uint8_t  switch_state;         /**< Power switch bitmask            */
} eps_telemetry_t;

/* ── Power switch channels ─────────────────────────────────────────────── */
typedef enum {
    EPS_CH_PAYLOAD   = 0,
    EPS_CH_ADCS      = 1,
    EPS_CH_COMMS     = 2,
    EPS_CH_HEATER_A  = 3,
    EPS_CH_HEATER_B  = 4,
    EPS_CH_SPARE_1   = 5,
    EPS_CH_SPARE_2   = 6,
    EPS_CH_SPARE_3   = 7
} eps_channel_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the EPS interface.
 * @return EPS_OK on success.
 */
int32_t eps_init(void);

/**
 * @brief Request current EPS telemetry via CAN.
 *
 * Sends SDO read request and parses response.
 *
 * @param[out] tlm  Telemetry output structure.
 * @return EPS_OK on success.
 */
int32_t eps_get_telemetry(eps_telemetry_t *tlm);

/**
 * @brief Control a power switch channel.
 *
 * @param[in] channel  Switch channel.
 * @param[in] on       1 = ON, 0 = OFF.
 * @return EPS_OK on success.
 */
int32_t eps_set_switch(eps_channel_t channel, uint8_t on);

/**
 * @brief Get current state of a power switch.
 *
 * @param[in] channel  Switch channel.
 * @return 1 if ON, 0 if OFF, negative on error.
 */
int32_t eps_get_switch(eps_channel_t channel);

/**
 * @brief Check if battery is in safe range.
 *
 * @return 1 if battery OK, 0 if low/critical.
 */
int32_t eps_battery_ok(void);

/**
 * @brief Periodic EPS tick — poll telemetry and check thresholds.
 *
 * Call every major frame (1 s).
 *
 * @return EPS_OK on success, negative on comm failure.
 */
int32_t eps_tick(void);

#ifdef __cplusplus
}
#endif

#endif /* EPS_INTERFACE_H */
