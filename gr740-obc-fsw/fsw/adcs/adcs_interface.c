/**
 * @file adcs_interface.c
 * @brief ADCS Interface implementation — CAN bus communication.
 *
 * Uses CANopen SDO for command/telemetry with ADCS node (CAN ID 3).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "adcs_interface.h"
#include "../../drivers/can/grcan.h"
#include "../../drivers/can/can_protocol.h"

extern int32_t  fdir_report_fault(uint16_t err_code, uint32_t aux);
extern uint32_t bsp_get_uptime_ms(void);

/* Error codes */
#define LOCAL_ERR_ADCS_SENSOR   0x0400U
#define LOCAL_ERR_ADCS_TUMBLING 0x0404U

/* ── CANopen SDO object indices for ADCS ───────────────────────────────── */
#define ADCS_SDO_QUAT_Q0    0x3000U
#define ADCS_SDO_QUAT_Q1    0x3001U
#define ADCS_SDO_QUAT_Q2    0x3002U
#define ADCS_SDO_QUAT_Q3    0x3003U
#define ADCS_SDO_OMEGA_X    0x3010U
#define ADCS_SDO_OMEGA_Y    0x3011U
#define ADCS_SDO_OMEGA_Z    0x3012U
#define ADCS_SDO_MODE       0x3100U
#define ADCS_SDO_STATUS     0x3101U
#define ADCS_SDO_SUN_ANG    0x3102U
#define ADCS_SDO_MODE_CMD   0x3200U

/* SDO COB-IDs */
#define ADCS_SDO_TX   (0x600U + CAN_NODE_ADCS)
#define ADCS_SDO_RX   (0x580U + CAN_NODE_ADCS)

/* ── Module state ──────────────────────────────────────────────────────── */
static adcs_telemetry_t last_tlm;
static adcs_mode_t      current_mode = ADCS_MODE_OFF;
static uint8_t          adcs_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

static int32_t sdo_read_i16(uint16_t index, uint8_t sub, int16_t *val)
{
    can_frame_t tx, rx;
    int32_t ret;
    uint32_t start;

    tx.id      = ADCS_SDO_TX;
    tx.dlc     = 8U;
    tx.rtr     = 0U;
    tx.data[0] = 0x40U;
    tx.data[1] = (uint8_t)(index & 0xFFU);
    tx.data[2] = (uint8_t)((index >> 8U) & 0xFFU);
    tx.data[3] = sub;
    tx.data[4] = 0U; tx.data[5] = 0U;
    tx.data[6] = 0U; tx.data[7] = 0U;

    ret = grcan_transmit(0U, &tx);
    if (ret != 0) { return ADCS_ERR_COMM; }

    start = bsp_get_uptime_ms();
    for (;;) {
        ret = grcan_receive(0U, &rx);
        if ((ret == 0) && (rx.id == ADCS_SDO_RX)) {
            *val = (int16_t)((uint16_t)rx.data[4] |
                             ((uint16_t)rx.data[5] << 8U));
            return ADCS_OK;
        }
        if ((bsp_get_uptime_ms() - start) > 100U) {
            return ADCS_ERR_TIMEOUT;
        }
    }
}

static int32_t sdo_write_u8(uint16_t index, uint8_t sub, uint8_t val)
{
    can_frame_t tx;
    tx.id      = ADCS_SDO_TX;
    tx.dlc     = 8U;
    tx.rtr     = 0U;
    tx.data[0] = 0x2FU;
    tx.data[1] = (uint8_t)(index & 0xFFU);
    tx.data[2] = (uint8_t)((index >> 8U) & 0xFFU);
    tx.data[3] = sub;
    tx.data[4] = val;
    tx.data[5] = 0U; tx.data[6] = 0U; tx.data[7] = 0U;

    return (grcan_transmit(0U, &tx) == 0) ? ADCS_OK : ADCS_ERR_COMM;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t adcs_init(void)
{
    last_tlm.q0 = 10000;  /* Identity quaternion */
    last_tlm.q1 = 0;
    last_tlm.q2 = 0;
    last_tlm.q3 = 0;
    last_tlm.omega_x = 0;
    last_tlm.omega_y = 0;
    last_tlm.omega_z = 0;
    last_tlm.mode = ADCS_MODE_OFF;
    last_tlm.pointing_ok = 0U;
    last_tlm.tumbling = 0U;
    last_tlm.sun_angle_01deg = 0;

    current_mode = ADCS_MODE_OFF;
    adcs_init_done = 1U;
    return ADCS_OK;
}

int32_t adcs_get_telemetry(adcs_telemetry_t *tlm)
{
    int32_t ret;
    int16_t tmp;
    uint16_t status;

    if (tlm == (adcs_telemetry_t *)0) {
        return ADCS_ERR_PARAM;
    }

    /* Quaternion */
    ret = sdo_read_i16(ADCS_SDO_QUAT_Q0, 0U, &tlm->q0);
    if (ret != ADCS_OK) { return ret; }
    ret = sdo_read_i16(ADCS_SDO_QUAT_Q1, 0U, &tlm->q1);
    if (ret != ADCS_OK) { return ret; }
    ret = sdo_read_i16(ADCS_SDO_QUAT_Q2, 0U, &tlm->q2);
    if (ret != ADCS_OK) { return ret; }
    ret = sdo_read_i16(ADCS_SDO_QUAT_Q3, 0U, &tlm->q3);
    if (ret != ADCS_OK) { return ret; }

    /* Angular rates */
    ret = sdo_read_i16(ADCS_SDO_OMEGA_X, 0U, &tlm->omega_x);
    if (ret != ADCS_OK) { return ret; }
    ret = sdo_read_i16(ADCS_SDO_OMEGA_Y, 0U, &tlm->omega_y);
    if (ret != ADCS_OK) { return ret; }
    ret = sdo_read_i16(ADCS_SDO_OMEGA_Z, 0U, &tlm->omega_z);
    if (ret != ADCS_OK) { return ret; }

    /* Mode and status */
    ret = sdo_read_i16(ADCS_SDO_MODE, 0U, &tmp);
    if (ret != ADCS_OK) { return ret; }
    tlm->mode = (adcs_mode_t)(tmp & 0xFFU);

    ret = sdo_read_i16(ADCS_SDO_STATUS, 0U, &tmp);
    if (ret != ADCS_OK) { return ret; }
    status = (uint16_t)tmp;
    tlm->pointing_ok = (uint8_t)(status & 0x01U);
    tlm->tumbling    = (uint8_t)((status >> 1U) & 0x01U);

    ret = sdo_read_i16(ADCS_SDO_SUN_ANG, 0U, &tlm->sun_angle_01deg);
    if (ret != ADCS_OK) { return ret; }

    last_tlm = *tlm;
    current_mode = tlm->mode;

    return ADCS_OK;
}

int32_t adcs_set_mode(adcs_mode_t mode)
{
    int32_t ret = sdo_write_u8(ADCS_SDO_MODE_CMD, 0U, (uint8_t)mode);
    if (ret == ADCS_OK) {
        current_mode = mode;
    }
    return ret;
}

adcs_mode_t adcs_get_mode(void)
{
    return current_mode;
}

int32_t adcs_is_tumbling(void)
{
    return (int32_t)last_tlm.tumbling;
}

int32_t adcs_tick(void)
{
    adcs_telemetry_t tlm;
    int32_t ret;

    if (adcs_init_done == 0U) {
        return ADCS_ERR_PARAM;
    }

    ret = adcs_get_telemetry(&tlm);
    if (ret != ADCS_OK) {
        (void)fdir_report_fault(LOCAL_ERR_ADCS_SENSOR, 0U);
        return ret;
    }

    if (tlm.tumbling != 0U) {
        (void)fdir_report_fault(LOCAL_ERR_ADCS_TUMBLING, 0U);
    }

    return ADCS_OK;
}
