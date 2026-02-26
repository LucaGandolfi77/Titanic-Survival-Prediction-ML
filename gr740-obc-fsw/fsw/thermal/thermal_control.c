/**
 * @file thermal_control.c
 * @brief Thermal Control Subsystem implementation.
 *
 * Bang-bang controller with hysteresis for each zone.
 * Sensors read via I2C (TMP175-compatible, 12-bit).
 * Heaters driven via GPIO outputs.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "thermal_control.h"
#include "../../drivers/i2c/i2cmst.h"
#include "../../drivers/gpio/grgpio.h"

/* Forward declarations for FDIR integration */
extern int32_t fdir_report_fault(uint16_t err_code, uint32_t aux);

/* Error codes from fdir/error_codes.h */
#define LOCAL_ERR_THERM_OVERTEMP   0x0600U
#define LOCAL_ERR_THERM_UNDERTEMP  0x0601U
#define LOCAL_ERR_THERM_SENSOR     0x0602U

/* ── I2C sensor configuration ─────────────────────────────────────────── */
/* TMP175 compatible: 7-bit addresses 0x48..0x4F, temperature register 0x00,
 * 12-bit resolution, data format: [MSB][LSB], temp = raw >> 4, 0.0625°C/LSB */
#define THERM_I2C_CTRL          0U
#define THERM_SENSOR_BASE_ADDR  0x48U
#define THERM_TEMP_REG          0x00U

/* ── Zone state ────────────────────────────────────────────────────────── */
typedef struct {
    therm_zone_config_t cfg;
    int16_t             current_temp;
    uint8_t             heater_on;
    uint8_t             override;       /* 1 = manual override active */
    uint8_t             override_state; /* heater state during override */
    uint8_t             configured;
} therm_zone_t;

/* ── Module state ──────────────────────────────────────────────────────── */
static therm_zone_t zones[THERM_MAX_ZONES];
static uint8_t      therm_init_done = 0U;

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t thermal_init(void)
{
    uint32_t i;

    for (i = 0U; i < THERM_MAX_ZONES; i++) {
        zones[i].current_temp   = 0;
        zones[i].heater_on      = 0U;
        zones[i].override       = 0U;
        zones[i].override_state = 0U;
        zones[i].configured     = 0U;
    }

    therm_init_done = 1U;
    return THERM_OK;
}

int32_t thermal_configure_zone(uint8_t zone_id,
                                const therm_zone_config_t *config)
{
    if (zone_id >= THERM_MAX_ZONES) {
        return THERM_ERR_PARAM;
    }
    if (config == (const therm_zone_config_t *)0) {
        return THERM_ERR_PARAM;
    }

    zones[zone_id].cfg        = *config;
    zones[zone_id].configured = 1U;

    /* Configure heater GPIO pin as output, initially OFF */
    gpio_set_direction(0U, config->heater_gpio_pin, GPIO_DIR_OUTPUT);
    gpio_write_pin(0U, config->heater_gpio_pin, 0U);

    return THERM_OK;
}

int32_t thermal_read_sensor(uint8_t sensor_id, int16_t *temp)
{
    uint8_t  i2c_addr;
    uint8_t  raw[2];
    int16_t  raw16;
    int32_t  ret;

    if (sensor_id >= THERM_MAX_SENSORS) {
        return THERM_ERR_PARAM;
    }
    if (temp == (int16_t *)0) {
        return THERM_ERR_PARAM;
    }

    i2c_addr = (uint8_t)(THERM_SENSOR_BASE_ADDR + sensor_id);

    /* Read 2 bytes from temperature register */
    ret = i2c_read_reg(THERM_I2C_CTRL, i2c_addr, THERM_TEMP_REG,
                        raw, 2U);
    if (ret != 0) {
        return THERM_ERR_SENSOR;
    }

    /* Parse 12-bit two's complement:
     * raw16 = (raw[0] << 8 | raw[1]) >> 4
     * Resolution: 0.0625 °C per LSB
     * Convert to 0.1 °C: temp = raw16 * 0.0625 * 10 = raw16 * 625 / 1000 */
    raw16 = (int16_t)(((int16_t)((int8_t)raw[0]) << 8) | (int16_t)raw[1]);
    raw16 >>= 4;

    /* Convert to 0.1 °C */
    *temp = (int16_t)((int32_t)raw16 * 625 / 1000);

    return THERM_OK;
}

int32_t thermal_tick(void)
{
    uint32_t i;
    int32_t  heaters_on = 0;
    int16_t  temp;
    int32_t  ret;

    if (therm_init_done == 0U) {
        return 0;
    }

    for (i = 0U; i < THERM_MAX_ZONES; i++) {
        if (zones[i].configured == 0U) {
            continue;
        }

        /* Read sensor */
        ret = thermal_read_sensor(zones[i].cfg.sensor_id, &temp);
        if (ret != THERM_OK) {
            (void)fdir_report_fault(LOCAL_ERR_THERM_SENSOR, (uint32_t)i);
            continue;
        }

        zones[i].current_temp = temp;

        /* Check alarms */
        if (temp > zones[i].cfg.alarm_high) {
            (void)fdir_report_fault(LOCAL_ERR_THERM_OVERTEMP,
                                     (uint32_t)(uint16_t)temp);
        }
        if (temp < zones[i].cfg.alarm_low) {
            (void)fdir_report_fault(LOCAL_ERR_THERM_UNDERTEMP,
                                     (uint32_t)(uint16_t)temp);
        }

        /* Manual override check */
        if (zones[i].override != 0U) {
            zones[i].heater_on = zones[i].override_state;
        } else {
            /* Bang-bang with hysteresis */
            if (zones[i].heater_on != 0U) {
                /* Heater is ON — turn OFF when above setpoint + hysteresis */
                if (temp >= (zones[i].cfg.set_point + zones[i].cfg.hysteresis)) {
                    zones[i].heater_on = 0U;
                }
            } else {
                /* Heater is OFF — turn ON when below setpoint - hysteresis */
                if (temp <= (zones[i].cfg.set_point - zones[i].cfg.hysteresis)) {
                    zones[i].heater_on = 1U;
                }
            }
        }

        /* Drive GPIO */
        gpio_write_pin(0U, zones[i].cfg.heater_gpio_pin, zones[i].heater_on);

        if (zones[i].heater_on != 0U) {
            heaters_on++;
        }
    }

    return heaters_on;
}

int32_t thermal_get_zone_temp(uint8_t zone_id, int16_t *temp)
{
    if (zone_id >= THERM_MAX_ZONES) {
        return THERM_ERR_PARAM;
    }
    if (temp == (int16_t *)0) {
        return THERM_ERR_PARAM;
    }
    if (zones[zone_id].configured == 0U) {
        return THERM_ERR_PARAM;
    }

    *temp = zones[zone_id].current_temp;
    return THERM_OK;
}

int32_t thermal_get_heater_state(uint8_t zone_id)
{
    if (zone_id >= THERM_MAX_ZONES) {
        return THERM_ERR_PARAM;
    }
    if (zones[zone_id].configured == 0U) {
        return THERM_ERR_PARAM;
    }

    return (int32_t)zones[zone_id].heater_on;
}

int32_t thermal_set_heater_override(uint8_t zone_id, uint8_t on)
{
    if (zone_id >= THERM_MAX_ZONES) {
        return THERM_ERR_PARAM;
    }
    if (zones[zone_id].configured == 0U) {
        return THERM_ERR_PARAM;
    }

    zones[zone_id].override       = 1U;
    zones[zone_id].override_state = (on != 0U) ? 1U : 0U;
    return THERM_OK;
}
