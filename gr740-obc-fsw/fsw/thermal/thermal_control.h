/**
 * @file thermal_control.h
 * @brief Thermal Control Subsystem API.
 *
 * Reads temperature sensors via I2C/SPI and controls heaters
 * via GPIO with hysteresis-based bang-bang control.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef THERMAL_CONTROL_H
#define THERMAL_CONTROL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define THERM_OK            0
#define THERM_ERR_PARAM    -1
#define THERM_ERR_SENSOR   -2

/* ── Configuration ─────────────────────────────────────────────────────── */
#define THERM_MAX_ZONES     8U    /**< Max thermal zones                 */
#define THERM_MAX_SENSORS  16U    /**< Max temperature sensors           */

/* ── Temperature units ─────────────────────────────────────────────────── */
/* Temperatures stored in 0.1 °C resolution as int16_t.
 * E.g. 25.3 °C = 253, -10.0 °C = -100.                                */

/* ── Zone descriptor ───────────────────────────────────────────────────── */
typedef struct {
    int16_t  set_point;        /**< Target temp (0.1°C)                  */
    int16_t  hysteresis;       /**< Hysteresis band (0.1°C)              */
    int16_t  alarm_high;       /**< Over-temperature alarm (0.1°C)       */
    int16_t  alarm_low;        /**< Under-temperature alarm (0.1°C)      */
    uint8_t  sensor_id;        /**< Primary sensor for this zone         */
    uint8_t  heater_gpio_pin;  /**< GPIO pin controlling heater          */
} therm_zone_config_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the thermal control subsystem.
 * @return THERM_OK on success.
 */
int32_t thermal_init(void);

/**
 * @brief Configure a thermal zone.
 *
 * @param[in] zone_id   Zone identifier (0..THERM_MAX_ZONES-1).
 * @param[in] config    Zone configuration.
 * @return THERM_OK on success.
 */
int32_t thermal_configure_zone(uint8_t zone_id,
                                const therm_zone_config_t *config);

/**
 * @brief Read a temperature sensor.
 *
 * @param[in]  sensor_id  Sensor index.
 * @param[out] temp       Temperature in 0.1°C.
 * @return THERM_OK on success, THERM_ERR_SENSOR on read failure.
 */
int32_t thermal_read_sensor(uint8_t sensor_id, int16_t *temp);

/**
 * @brief Periodic thermal control tick.
 *
 * Reads all zone sensors, applies bang-bang control, checks alarms.
 * Call every 1 s.
 *
 * @return Number of heaters currently ON.
 */
int32_t thermal_tick(void);

/**
 * @brief Get current temperature of a zone.
 *
 * @param[in]  zone_id  Zone identifier.
 * @param[out] temp     Current temperature (0.1°C).
 * @return THERM_OK on success.
 */
int32_t thermal_get_zone_temp(uint8_t zone_id, int16_t *temp);

/**
 * @brief Get heater state for a zone.
 *
 * @param[in] zone_id  Zone identifier.
 * @return 1 if heater ON, 0 if OFF, negative on error.
 */
int32_t thermal_get_heater_state(uint8_t zone_id);

/**
 * @brief Force heater ON/OFF (manual override).
 *
 * @param[in] zone_id  Zone identifier.
 * @param[in] on       1 = ON, 0 = OFF.
 * @return THERM_OK on success.
 */
int32_t thermal_set_heater_override(uint8_t zone_id, uint8_t on);

#ifdef __cplusplus
}
#endif

#endif /* THERMAL_CONTROL_H */
