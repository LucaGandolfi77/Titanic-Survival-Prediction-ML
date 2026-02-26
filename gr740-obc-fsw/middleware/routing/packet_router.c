/**
 * @file packet_router.c
 * @brief Packet Router — TC dispatch and TM routing implementation.
 *
 * Routes incoming TCs from any interface to PUS service handlers.
 * Queues outbound TM and flushes via the active downlink path.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "packet_router.h"
#include "../ccsds/space_packet.h"

/* ── Constants ─────────────────────────────────────────────────────────── */
#define PUS_SEC_HDR_MIN_LEN  9U    /* ver + svc + sub + srcid(2) + time(4) */
#define CCSDS_HDR_LEN        6U

/* ── Service handler table ─────────────────────────────────────────────── */
typedef struct {
    uint8_t              svc_type;
    router_tc_handler_t  handler;
    uint8_t              registered;
} svc_entry_t;

/* ── TM queue entry ────────────────────────────────────────────────────── */
typedef struct {
    uint8_t  data[CCSDS_MAX_PKT_SIZE];
    uint16_t len;
    uint8_t  used;
} tm_queue_entry_t;

/* ── Module state ──────────────────────────────────────────────────────── */
static svc_entry_t      svc_table[ROUTER_MAX_SERVICES];
static uint32_t         svc_count;

static tm_queue_entry_t tm_queue[ROUTER_TM_QUEUE_DEPTH];
static uint32_t         tm_head;
static uint32_t         tm_tail;
static uint32_t         tm_used_count;

static router_downlink_fn_t  dl_funcs[ROUTER_DL_COUNT];
static router_downlink_t     active_dl;

static uint32_t stat_tc;
static uint32_t stat_tm;
static uint32_t stat_err;

static uint8_t  router_initialised = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

/**
 * @brief Extract 16-bit value from big-endian buffer.
 */
static uint16_t get_u16(const uint8_t *buf)
{
    return (uint16_t)(((uint16_t)buf[0] << 8U) | (uint16_t)buf[1]);
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t router_init(void)
{
    uint32_t i;

    for (i = 0U; i < ROUTER_MAX_SERVICES; i++) {
        svc_table[i].svc_type   = 0U;
        svc_table[i].handler    = (router_tc_handler_t)0;
        svc_table[i].registered = 0U;
    }
    svc_count = 0U;

    for (i = 0U; i < ROUTER_TM_QUEUE_DEPTH; i++) {
        tm_queue[i].len  = 0U;
        tm_queue[i].used = 0U;
    }
    tm_head       = 0U;
    tm_tail       = 0U;
    tm_used_count = 0U;

    for (i = 0U; i < (uint32_t)ROUTER_DL_COUNT; i++) {
        dl_funcs[i] = (router_downlink_fn_t)0;
    }
    active_dl = ROUTER_DL_SPW;

    stat_tc  = 0U;
    stat_tm  = 0U;
    stat_err = 0U;

    router_initialised = 1U;
    return ROUTER_OK;
}

int32_t router_register_service(uint8_t svc_type, router_tc_handler_t handler)
{
    uint32_t i;

    if ((handler == (router_tc_handler_t)0) || (svc_type == 0U)) {
        return ROUTER_ERR_PARAM;
    }

    /* Check for existing registration — update */
    for (i = 0U; i < svc_count; i++) {
        if ((svc_table[i].registered != 0U) &&
            (svc_table[i].svc_type == svc_type)) {
            svc_table[i].handler = handler;
            return ROUTER_OK;
        }
    }

    /* New registration */
    if (svc_count >= ROUTER_MAX_SERVICES) {
        return ROUTER_ERR_QUEUE_FULL;
    }

    svc_table[svc_count].svc_type   = svc_type;
    svc_table[svc_count].handler    = handler;
    svc_table[svc_count].registered = 1U;
    svc_count++;

    return ROUTER_OK;
}

int32_t router_register_downlink(router_downlink_t dl_path,
                                  router_downlink_fn_t fn)
{
    if ((dl_path >= ROUTER_DL_COUNT) || (fn == (router_downlink_fn_t)0)) {
        return ROUTER_ERR_PARAM;
    }

    dl_funcs[dl_path] = fn;
    return ROUTER_OK;
}

int32_t router_set_active_downlink(router_downlink_t dl_path)
{
    if (dl_path >= ROUTER_DL_COUNT) {
        return ROUTER_ERR_PARAM;
    }

    active_dl = dl_path;
    return ROUTER_OK;
}

int32_t router_dispatch_tc(const uint8_t *raw, uint16_t len)
{
    ccsds_packet_t  pkt;
    uint16_t        pkt_id;
    uint16_t        data_len;
    uint16_t        crc_rx;
    uint16_t        crc_calc;
    const uint8_t  *data_field;
    uint8_t         svc_type;
    uint8_t         svc_subtype;
    uint16_t        tc_data_len;
    uint32_t        i;
    int32_t         ret;

    if ((raw == (const uint8_t *)0) || (len < (CCSDS_HDR_LEN + 2U))) {
        stat_err++;
        return ROUTER_ERR_PARAM;
    }

    /* Deserialise CCSDS header */
    ret = ccsds_deserialize(&pkt, raw, len);
    if (ret != CCSDS_OK) {
        stat_err++;
        return ROUTER_ERR_CRC;
    }

    /* Verify CRC-16 */
    data_len = pkt.header.pkt_data_len + 1U;  /* data field length */
    if ((CCSDS_HDR_LEN + data_len) > len) {
        stat_err++;
        return ROUTER_ERR_PARAM;
    }

    crc_rx = get_u16(&raw[CCSDS_HDR_LEN + data_len - 2U]);
    crc_calc = ccsds_crc16(raw, (uint16_t)(CCSDS_HDR_LEN + data_len - 2U));
    if (crc_rx != crc_calc) {
        stat_err++;
        return ROUTER_ERR_CRC;
    }

    /* Verify it's a TC (type = 1) */
    pkt_id = pkt.header.pkt_id;
    if (((pkt_id >> 12U) & 0x01U) != 1U) {
        stat_err++;
        return ROUTER_ERR_PARAM;
    }

    /* Extract PUS secondary header */
    data_field = pkt.data;
    if (pkt.data_len < PUS_SEC_HDR_MIN_LEN) {
        stat_err++;
        return ROUTER_ERR_PARAM;
    }

    /* Byte 0: PUS version | Byte 1: svc_type | Byte 2: svc_subtype */
    svc_type    = data_field[1];
    svc_subtype = data_field[2];

    /* TC application data follows the 9-byte PUS secondary header */
    tc_data_len = (pkt.data_len > (PUS_SEC_HDR_MIN_LEN + 2U))
                  ? (uint16_t)(pkt.data_len - PUS_SEC_HDR_MIN_LEN - 2U)
                  : 0U;  /* subtract CRC (2) and PUS secondary header */

    /* Look up handler */
    for (i = 0U; i < svc_count; i++) {
        if ((svc_table[i].registered != 0U) &&
            (svc_table[i].svc_type == svc_type)) {
            ret = svc_table[i].handler(svc_subtype,
                                        &data_field[PUS_SEC_HDR_MIN_LEN],
                                        tc_data_len);
            stat_tc++;
            return (ret == 0) ? ROUTER_OK : ret;
        }
    }

    stat_err++;
    return ROUTER_ERR_NO_HANDLER;
}

int32_t router_send_tm(const ccsds_packet_t *pkt)
{
    uint8_t  buf[CCSDS_MAX_PKT_SIZE];
    uint16_t ser_len;
    int32_t  ret;

    if (pkt == (const ccsds_packet_t *)0) {
        return ROUTER_ERR_PARAM;
    }
    if (router_initialised == 0U) {
        return ROUTER_ERR_PARAM;
    }

    /* Serialise */
    ret = ccsds_serialize(pkt, buf, CCSDS_MAX_PKT_SIZE, &ser_len);
    if (ret != CCSDS_OK) {
        stat_err++;
        return ROUTER_ERR_PARAM;
    }

    /* Enqueue */
    if (tm_used_count >= ROUTER_TM_QUEUE_DEPTH) {
        stat_err++;
        return ROUTER_ERR_QUEUE_FULL;
    }

    {
        tm_queue_entry_t *entry = &tm_queue[tm_head];
        uint16_t j;
        for (j = 0U; j < ser_len; j++) {
            entry->data[j] = buf[j];
        }
        entry->len  = ser_len;
        entry->used = 1U;
    }

    tm_head = (tm_head + 1U) % ROUTER_TM_QUEUE_DEPTH;
    tm_used_count++;
    stat_tm++;

    return ROUTER_OK;
}

int32_t router_process_tm_queue(void)
{
    int32_t  sent = 0;
    int32_t  ret;

    if (router_initialised == 0U) {
        return ROUTER_ERR_PARAM;
    }

    while (tm_used_count > 0U) {
        tm_queue_entry_t *entry = &tm_queue[tm_tail];

        if (entry->used == 0U) {
            break;
        }

        /* Send via active downlink */
        if (dl_funcs[active_dl] != (router_downlink_fn_t)0) {
            ret = dl_funcs[active_dl](entry->data, entry->len);
            if (ret < 0) {
                stat_err++;
                return ROUTER_ERR_DOWNLINK;
            }
        }

        entry->used = 0U;
        tm_tail = (tm_tail + 1U) % ROUTER_TM_QUEUE_DEPTH;
        tm_used_count--;
        sent++;
    }

    return sent;
}

void router_get_stats(uint32_t *tc_count, uint32_t *tm_count,
                      uint32_t *err_count)
{
    if (tc_count != (uint32_t *)0) {
        *tc_count = stat_tc;
    }
    if (tm_count != (uint32_t *)0) {
        *tm_count = stat_tm;
    }
    if (err_count != (uint32_t *)0) {
        *err_count = stat_err;
    }
}
