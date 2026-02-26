/**
 * @file eps_interface.c
 * @brief EPS Interface implementation — CAN bus communication.
 *
 * Uses CANopen SDO expedited transfers (4-byte data).
 * EPS node address: CAN node ID 2 (from can_protocol.h).
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "eps_interface.h"
#include "../../drivers/can/grcan.h"
#include "../../drivers/can/can_protocol.h"

/* Forward declaration for FDIR */
extern int32_t fdir_report_fault(uint16_t err_code, uint32_t aux);
extern uint32_t bsp_get_uptime_ms(void);

/* Error codes */
#define LOCAL_ERR_EPS_UNDERVOLT  0x0500U
#define LOCAL_ERR_EPS_BATTERY    0x0503U
#define LOCAL_ERR_EPS_COMM       0x0506U

/* ── CANopen SDO object indices for EPS ────────────────────────────────── */
#define EPS_SDO_BAT_VOLT     0x2100U
#define EPS_SDO_BAT_CURR     0x2101U
#define EPS_SDO_BAT_SOC      0x2102U
#define EPS_SDO_BAT_TEMP     0x2103U
#define EPS_SDO_SOL_VOLT     0x2110U
#define EPS_SDO_SOL_CURR     0x2111U
#define EPS_SDO_BUS_VOLT     0x2120U
#define EPS_SDO_SW_STATE     0x2200U
#define EPS_SDO_SW_CTRL      0x2201U

/* SDO COB-IDs */
#define SDO_TX_COBID  (0x600U + CAN_NODE_EPS)  /* OBC → EPS */
#define SDO_RX_COBID  (0x580U + CAN_NODE_EPS)  /* EPS → OBC */

/* SDO command specifiers */
#define SDO_READ_CMD   0x40U
#define SDO_WRITE_CMD  0x23U  /* 4-byte expedited write */

/* Thresholds */
#define EPS_BAT_LOW_MV     6400U    /**< 6.4V low threshold  */
#define EPS_BAT_CRIT_MV    6000U    /**< 6.0V critical       */
#define EPS_BUS_LOW_MV     4800U    /**< 4.8V bus undervolt   */

/* ── Module state ──────────────────────────────────────────────────────── */
static eps_telemetry_t  last_tlm;
static uint8_t          switch_state_cache = 0U;
static uint8_t          eps_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

/**
 * @brief Send SDO read request and receive 4-byte response.
 */
static int32_t sdo_read_u16(uint16_t index, uint8_t subindex, uint16_t *val)
{
    can_frame_t tx_frame;
    can_frame_t rx_frame;
    int32_t     ret;
    uint32_t    timeout;
    uint32_t    start;

    /* Build SDO upload request */
    tx_frame.id     = SDO_TX_COBID;
    tx_frame.dlc    = 8U;
    tx_frame.rtr    = 0U;
    tx_frame.data[0] = SDO_READ_CMD;
    tx_frame.data[1] = (uint8_t)(index & 0xFFU);
    tx_frame.data[2] = (uint8_t)((index >> 8U) & 0xFFU);
    tx_frame.data[3] = subindex;
    tx_frame.data[4] = 0U;
    tx_frame.data[5] = 0U;
    tx_frame.data[6] = 0U;
    tx_frame.data[7] = 0U;

    ret = grcan_transmit(0U, &tx_frame);
    if (ret != 0) {
        return EPS_ERR_COMM;
    }

    /* Wait for response with timeout */
    start = bsp_get_uptime_ms();
    timeout = 100U;  /* 100 ms */

    for (;;) {
        ret = grcan_receive(0U, &rx_frame);
        if (ret == 0) {
            if (rx_frame.id == SDO_RX_COBID) {
                /* Parse expedited response */
                *val = (uint16_t)((uint16_t)rx_frame.data[4] |
                                  ((uint16_t)rx_frame.data[5] << 8U));
                return EPS_OK;
            }
        }
        if ((bsp_get_uptime_ms() - start) > timeout) {
            return EPS_ERR_TIMEOUT;
        }
    }
}

static int32_t sdo_write_u8(uint16_t index, uint8_t subindex, uint8_t val)
{
    can_frame_t tx_frame;
    int32_t     ret;

    tx_frame.id     = SDO_TX_COBID;
    tx_frame.dlc    = 8U;
    tx_frame.rtr    = 0U;
    tx_frame.data[0] = 0x2FU;  /* Expedited, 1 byte */
    tx_frame.data[1] = (uint8_t)(index & 0xFFU);
    tx_frame.data[2] = (uint8_t)((index >> 8U) & 0xFFU);
    tx_frame.data[3] = subindex;
    tx_frame.data[4] = val;
    tx_frame.data[5] = 0U;
    tx_frame.data[6] = 0U;
    tx_frame.data[7] = 0U;

    ret = grcan_transmit(0U, &tx_frame);
    return (ret == 0) ? EPS_OK : EPS_ERR_COMM;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t eps_init(void)
{
    last_tlm.battery_voltage_mv = 0U;
    last_tlm.battery_current_ma = 0;
    last_tlm.battery_soc_pct    = 0U;
    last_tlm.battery_temp_01c   = 0;
    last_tlm.solar_voltage_mv   = 0U;
    last_tlm.solar_current_ma   = 0;
    last_tlm.bus_voltage_mv     = 0U;
    last_tlm.switch_state       = 0U;

    switch_state_cache = 0U;
    eps_init_done = 1U;
    return EPS_OK;
}

int32_t eps_get_telemetry(eps_telemetry_t *tlm)
{
    int32_t  ret;
    uint16_t tmp;

    if (tlm == (eps_telemetry_t *)0) {
        return EPS_ERR_PARAM;
    }

    /* Read each parameter via SDO */
    ret = sdo_read_u16(EPS_SDO_BAT_VOLT, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->battery_voltage_mv = tmp;

    ret = sdo_read_u16(EPS_SDO_BAT_CURR, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->battery_current_ma = (int16_t)tmp;

    ret = sdo_read_u16(EPS_SDO_BAT_SOC, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->battery_soc_pct = (uint8_t)(tmp & 0xFFU);

    ret = sdo_read_u16(EPS_SDO_BAT_TEMP, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->battery_temp_01c = (int16_t)tmp;

    ret = sdo_read_u16(EPS_SDO_SOL_VOLT, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->solar_voltage_mv = tmp;

    ret = sdo_read_u16(EPS_SDO_SOL_CURR, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->solar_current_ma = (int16_t)tmp;

    ret = sdo_read_u16(EPS_SDO_BUS_VOLT, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->bus_voltage_mv = tmp;

    ret = sdo_read_u16(EPS_SDO_SW_STATE, 0U, &tmp);
    if (ret != EPS_OK) { return ret; }
    tlm->switch_state = (uint8_t)(tmp & 0xFFU);

    /* Cache */
    last_tlm = *tlm;
    switch_state_cache = tlm->switch_state;

    return EPS_OK;
}

int32_t eps_set_switch(eps_channel_t channel, uint8_t on)
{
    uint8_t mask;
    uint8_t new_state;

    if ((uint32_t)channel >= EPS_MAX_CHANNELS) {
        return EPS_ERR_PARAM;
    }

    mask = (uint8_t)(1U << (uint8_t)channel);
    if (on != 0U) {
        new_state = switch_state_cache | mask;
    } else {
        new_state = switch_state_cache & (uint8_t)(~mask);
    }

    /* Send via CAN SDO */
    {
        int32_t ret = sdo_write_u8(EPS_SDO_SW_CTRL, 0U, new_state);
        if (ret == EPS_OK) {
            switch_state_cache = new_state;
        }
        return ret;
    }
}

int32_t eps_get_switch(eps_channel_t channel)
{
    if ((uint32_t)channel >= EPS_MAX_CHANNELS) {
        return EPS_ERR_PARAM;
    }
    return ((switch_state_cache >> (uint8_t)channel) & 0x01U) != 0U ? 1 : 0;
}

int32_t eps_battery_ok(void)
{
    if (last_tlm.battery_voltage_mv < EPS_BAT_CRIT_MV) {
        return 0;
    }
    return 1;
}

int32_t eps_tick(void)
{
    eps_telemetry_t tlm;
    int32_t ret;

    if (eps_init_done == 0U) {
        return EPS_ERR_PARAM;
    }

    ret = eps_get_telemetry(&tlm);
    if (ret != EPS_OK) {
        (void)fdir_report_fault(LOCAL_ERR_EPS_COMM, 0U);
        return ret;
    }

    /* Check battery */
    if (tlm.battery_voltage_mv < EPS_BAT_LOW_MV) {
        (void)fdir_report_fault(LOCAL_ERR_EPS_BATTERY,
                                 (uint32_t)tlm.battery_voltage_mv);
    }

    /* Check bus voltage */
    if (tlm.bus_voltage_mv < EPS_BUS_LOW_MV) {
        (void)fdir_report_fault(LOCAL_ERR_EPS_UNDERVOLT,
                                 (uint32_t)tlm.bus_voltage_mv);
    }

    return EPS_OK;
}
