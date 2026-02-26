/**
 * @file pus_st09.c
 * @brief PUS Service 9 — Time Management implementation.
 *
 * Maintains on-board CUC time (4+2 format per CCSDS 301.0-B-4).
 * Coarse time = seconds since mission epoch.
 * Fine time = 1/65536 second resolution.
 * Updated by 1ms tick from systick ISR.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "pus_st09.h"
#include "../ccsds/space_packet.h"

extern int32_t router_send_tm(const ccsds_packet_t *pkt);

/* ── Module state ──────────────────────────────────────────────────────── */
static cuc_time_t obt;              /**< On-board time              */
static uint32_t   sub_ms_accum;     /**< Sub-ms accumulator (fine)  */
static uint16_t   st09_apid;
static uint8_t    st09_init_done = 0U;

/**
 * Fine counter increment per millisecond:
 * 65536 / 1000 = 65.536 → use fixed-point 65 + accumulator
 * To avoid drift: 65536 ticks / 1000 ms
 * Accumulate fractional part: 536 per ms, roll over at 1000
 */
#define FINE_PER_MS       65U
#define FINE_FRAC_PER_MS  536U
#define FINE_FRAC_LIMIT   1000U

#define PUS_VERSION_C    0x20U
#define PUS_SVC_TYPE_09  9U

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t pus_st09_init(uint16_t apid)
{
    if (apid > 0x7FFU) {
        return PUS_ST09_ERR_PARAM;
    }

    st09_apid = apid;
    obt.coarse = 0U;
    obt.fine = 0U;
    sub_ms_accum = 0U;
    st09_init_done = 1U;

    return PUS_ST09_OK;
}

int32_t pus_st09_set_time(const cuc_time_t *time)
{
    if (time == (const cuc_time_t *)0) {
        return PUS_ST09_ERR_PARAM;
    }

    obt.coarse = time->coarse;
    obt.fine = time->fine;
    sub_ms_accum = 0U;

    return PUS_ST09_OK;
}

int32_t pus_st09_get_time(cuc_time_t *time)
{
    if (time == (cuc_time_t *)0) {
        return PUS_ST09_ERR_PARAM;
    }

    time->coarse = obt.coarse;
    time->fine = obt.fine;

    return PUS_ST09_OK;
}

void pus_st09_tick_ms(void)
{
    uint32_t new_fine;

    if (st09_init_done == 0U) {
        return;
    }

    /* Advance fine counter */
    new_fine = (uint32_t)obt.fine + FINE_PER_MS;

    /* Handle fractional accumulation to avoid drift */
    sub_ms_accum += FINE_FRAC_PER_MS;
    if (sub_ms_accum >= FINE_FRAC_LIMIT) {
        sub_ms_accum -= FINE_FRAC_LIMIT;
        new_fine += 1U;
    }

    /* Check for second rollover */
    if (new_fine >= 65536U) {
        obt.coarse++;
        new_fine -= 65536U;
    }

    obt.fine = (uint16_t)new_fine;
}

int32_t pus_st09_report(void)
{
    ccsds_packet_t tm_pkt;
    uint8_t data[16]; /* PUS sec hdr + CUC time */
    uint32_t pos;
    uint16_t seq;
    int32_t  ret;

    if (st09_init_done == 0U) {
        return PUS_ST09_ERR_PARAM;
    }

    pos = 0U;

    /* PUS-C secondary header */
    data[pos] = PUS_VERSION_C;    pos++;
    data[pos] = PUS_SVC_TYPE_09;  pos++;
    data[pos] = PUS_ST09_REPORT_TIME; pos++;
    data[pos] = 0x00U;            pos++; /* Source ID high */
    data[pos] = 0x01U;            pos++; /* Source ID low  */
    /* Timestamp = current time itself */
    data[pos] = (uint8_t)((obt.coarse >> 24U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((obt.coarse >> 16U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((obt.coarse >> 8U)  & 0xFFU); pos++;
    data[pos] = (uint8_t)(obt.coarse & 0xFFU);          pos++;

    /* CUC Time report data: coarse(4) + fine(2) */
    data[pos] = (uint8_t)((obt.coarse >> 24U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((obt.coarse >> 16U) & 0xFFU); pos++;
    data[pos] = (uint8_t)((obt.coarse >> 8U)  & 0xFFU); pos++;
    data[pos] = (uint8_t)(obt.coarse & 0xFFU);          pos++;
    data[pos] = (uint8_t)((obt.fine >> 8U) & 0xFFU);    pos++;
    data[pos] = (uint8_t)(obt.fine & 0xFFU);             pos++;

    ret = ccsds_init_packet(&tm_pkt, CCSDS_TYPE_TM, st09_apid,
                             CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    if (ret != CCSDS_OK) {
        return PUS_ST09_ERR_PARAM;
    }

    seq = ccsds_next_seq_count(st09_apid);
    tm_pkt.header.pkt_seq_ctrl = (uint16_t)(
        (tm_pkt.header.pkt_seq_ctrl & 0xC000U) | (seq & 0x3FFFU)
    );

    ret = ccsds_set_data(&tm_pkt, data, pos);
    if (ret != CCSDS_OK) {
        return PUS_ST09_ERR_PARAM;
    }

    ret = router_send_tm(&tm_pkt);
    (void)ret;

    return PUS_ST09_OK;
}

int32_t pus_st09_process_tc(const uint8_t *data, uint32_t len)
{
    cuc_time_t new_time;

    if (data == (const uint8_t *)0) {
        return PUS_ST09_ERR_PARAM;
    }
    if (len < 6U) { /* coarse(4) + fine(2) */
        return PUS_ST09_ERR_PARAM;
    }

    new_time.coarse = ((uint32_t)data[0] << 24U) |
                      ((uint32_t)data[1] << 16U) |
                      ((uint32_t)data[2] << 8U) |
                      (uint32_t)data[3];
    new_time.fine   = (uint16_t)(((uint16_t)data[4] << 8U) |
                                  (uint16_t)data[5]);

    return pus_st09_set_time(&new_time);
}
