/**
 * @file error_codes.h
 * @brief System-wide error code definitions.
 *
 * Enum groups for all subsystem errors, used by FDIR and event reporting.
 * Error ranges partitioned by subsystem for clear fault isolation.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error code ranges ────────────────────────────────────────────────── */
/* 0x0000        : No error                                              */
/* 0x0100-0x01FF : OBC / Processor errors                                */
/* 0x0200-0x02FF : Memory errors                                         */
/* 0x0300-0x03FF : Communication errors                                  */
/* 0x0400-0x04FF : Attitude control errors                               */
/* 0x0500-0x05FF : Power / EPS errors                                    */
/* 0x0600-0x06FF : Thermal errors                                        */
/* 0x0700-0x07FF : Payload errors                                        */
/* 0x0800-0x08FF : Software / task errors                                */
/* 0x0900-0x09FF : PUS / protocol errors                                 */

typedef enum {
    /* ── No Error ─────────────────────────────────────────────────────── */
    ERR_NONE                        = 0x0000U,

    /* ── OBC / Processor (0x01xx) ──────────────────────────────────────── */
    ERR_OBC_WATCHDOG_TIMEOUT        = 0x0100U,
    ERR_OBC_TASK_OVERRUN            = 0x0101U,
    ERR_OBC_STACK_OVERFLOW          = 0x0102U,
    ERR_OBC_EXCEPTION               = 0x0103U,
    ERR_OBC_CPU_OVERTEMP            = 0x0104U,
    ERR_OBC_BOOT_FAIL               = 0x0105U,
    ERR_OBC_CLK_FAULT               = 0x0106U,

    /* ── Memory (0x02xx) ──────────────────────────────────────────────── */
    ERR_MEM_EDAC_SINGLE             = 0x0200U,
    ERR_MEM_EDAC_DOUBLE             = 0x0201U,
    ERR_MEM_SRAM_FAIL               = 0x0202U,
    ERR_MEM_EEPROM_WRITE_FAIL       = 0x0203U,
    ERR_MEM_MRAM_FAIL               = 0x0204U,
    ERR_MEM_PROM_CRC_FAIL           = 0x0205U,
    ERR_MEM_SCRUB_FAIL              = 0x0206U,

    /* ── Communication (0x03xx) ───────────────────────────────────────── */
    ERR_COMM_SPW_LINK_DOWN          = 0x0300U,
    ERR_COMM_SPW_TIMEOUT            = 0x0301U,
    ERR_COMM_SPW_PARITY             = 0x0302U,
    ERR_COMM_CAN_BUS_OFF            = 0x0310U,
    ERR_COMM_CAN_TX_FAIL            = 0x0311U,
    ERR_COMM_CAN_RX_OVERRUN         = 0x0312U,
    ERR_COMM_UART_OVERRUN           = 0x0320U,
    ERR_COMM_UART_FRAMING           = 0x0321U,
    ERR_COMM_RMAP_CRC               = 0x0330U,
    ERR_COMM_RMAP_TIMEOUT           = 0x0331U,
    ERR_COMM_TC_CRC_FAIL            = 0x0340U,
    ERR_COMM_TC_SEQ_GAP             = 0x0341U,

    /* ── ADCS (0x04xx) ────────────────────────────────────────────────── */
    ERR_ADCS_SENSOR_FAIL            = 0x0400U,
    ERR_ADCS_ACTUATOR_FAIL          = 0x0401U,
    ERR_ADCS_POINTING_ERR           = 0x0402U,
    ERR_ADCS_GYRO_SATURATED         = 0x0403U,
    ERR_ADCS_TUMBLING               = 0x0404U,

    /* ── EPS / Power (0x05xx) ─────────────────────────────────────────── */
    ERR_EPS_UNDERVOLT               = 0x0500U,
    ERR_EPS_OVERVOLT                = 0x0501U,
    ERR_EPS_OVERCURRENT             = 0x0502U,
    ERR_EPS_BATTERY_LOW             = 0x0503U,
    ERR_EPS_SOLAR_FAIL              = 0x0504U,
    ERR_EPS_BUS_FAULT               = 0x0505U,
    ERR_EPS_COMM_LOSS               = 0x0506U,

    /* ── Thermal (0x06xx) ─────────────────────────────────────────────── */
    ERR_THERM_OVERTEMP              = 0x0600U,
    ERR_THERM_UNDERTEMP             = 0x0601U,
    ERR_THERM_SENSOR_FAIL           = 0x0602U,
    ERR_THERM_HEATER_FAIL           = 0x0603U,

    /* ── Payload (0x07xx) ─────────────────────────────────────────────── */
    ERR_PLD_POWER_FAIL              = 0x0700U,
    ERR_PLD_DATA_ERROR              = 0x0701U,
    ERR_PLD_TIMEOUT                 = 0x0702U,

    /* ── Software / Tasks (0x08xx) ────────────────────────────────────── */
    ERR_SW_ASSERT_FAIL              = 0x0800U,
    ERR_SW_PARAM_OOR                = 0x0801U,
    ERR_SW_QUEUE_FULL               = 0x0802U,
    ERR_SW_SCHED_OVERRUN            = 0x0803U,
    ERR_SW_INIT_FAIL                = 0x0804U,

    /* ── PUS / Protocol (0x09xx) ──────────────────────────────────────── */
    ERR_PUS_ILLEGAL_APID            = 0x0900U,
    ERR_PUS_ILLEGAL_TYPE            = 0x0901U,
    ERR_PUS_ILLEGAL_SUBTYPE         = 0x0902U,
    ERR_PUS_PKT_TOO_SHORT           = 0x0903U,
    ERR_PUS_CRC_FAIL                = 0x0904U,
    ERR_PUS_EXEC_FAIL               = 0x0905U

} error_code_t;

/**
 * @brief Get error code severity (0=info, 1=low, 2=medium, 3=high).
 *
 * Determined by range:
 *   0x02xx memory = high
 *   0x03xx comms  = medium
 *   0x04xx ADCS   = medium (tumbling = high)
 *   0x05xx EPS    = high
 *   default       = low
 */
static inline uint8_t error_get_severity(error_code_t code)
{
    uint16_t range = (uint16_t)code & 0xFF00U;

    switch (range) {
        case 0x0200U:  return 3U;  /* Memory — high    */
        case 0x0500U:  return 3U;  /* EPS — high       */
        case 0x0600U:  return 2U;  /* Thermal — medium */
        case 0x0300U:  return 2U;  /* Comms — medium   */
        case 0x0400U:  return 2U;  /* ADCS — medium    */
        case 0x0100U:  return 2U;  /* OBC — medium     */
        default:       return 1U;  /* Low              */
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ERROR_CODES_H */
