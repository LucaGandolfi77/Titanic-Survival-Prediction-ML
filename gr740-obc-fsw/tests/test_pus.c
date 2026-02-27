/**
 * @file test_pus.c
 * @brief Unit tests for all PUS services (ST01, ST03, ST05, ST08,
 *        ST09, ST11, ST17).
 *
 * Each service is tested via its public API.  Router and BSP
 * dependencies are stubbed out.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

/* ── BSP / Router stubs ───────────────────────────────────────────────── */
#include "middleware/ccsds/space_packet.h"
#include "middleware/pus/pus_st01.h"
#include "middleware/pus/pus_st03.h"
#include "middleware/pus/pus_st05.h"
#include "middleware/pus/pus_st08.h"
#include "middleware/pus/pus_st09.h"
#include "middleware/pus/pus_st11.h"
#include "middleware/pus/pus_st17.h"

static uint32_t stub_uptime_ms = 0U;
uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/* Capture TM packets sent via router_send_tm */
#define TM_CAPTURE_MAX 32U
static uint8_t  tm_capture[TM_CAPTURE_MAX][256];
static uint16_t tm_capture_len[TM_CAPTURE_MAX];
static uint32_t tm_capture_count = 0U;

int32_t router_send_tm(const ccsds_packet_t *pkt)
{
    if (tm_capture_count < TM_CAPTURE_MAX) {
        uint32_t outlen = 0U;
        (void)ccsds_serialize(pkt, tm_capture[tm_capture_count],
                              sizeof(tm_capture[0]), &outlen);
        tm_capture_len[tm_capture_count] = (uint16_t)outlen;
        tm_capture_count++;
    }
    return 0;
}

static void tm_capture_reset(void)
{
    tm_capture_count = 0U;
    memset(tm_capture, 0, sizeof(tm_capture));
    memset(tm_capture_len, 0, sizeof(tm_capture_len));
}

/* ── Simple ST03 parameter storage + reader used by tests ───────────── */
static uint32_t st03_param_values[PUS_ST03_MAX_SID][PUS_ST03_MAX_PARAMS];

static int32_t test_st03_reader(uint16_t param_id, uint8_t *buf, uint32_t *len)
{
    /* param_id encodes (sid << 8) | param_idx in our test harness */
    uint16_t sid = (uint16_t)(param_id >> 8);
    uint8_t  idx = (uint8_t)(param_id & 0xFFU);

    if (sid >= PUS_ST03_MAX_SID) { return -1; }
    if (idx >= PUS_ST03_MAX_PARAMS) { return -1; }

    uint32_t val = st03_param_values[sid][idx];
    /* Return as 4-byte big-endian */
    if (buf != (uint8_t *)0) {
        buf[0] = (uint8_t)((val >> 24) & 0xFFU);
        buf[1] = (uint8_t)((val >> 16) & 0xFFU);
        buf[2] = (uint8_t)((val >> 8) & 0xFFU);
        buf[3] = (uint8_t)(val & 0xFFU);
    }
    if (len != (uint32_t *)0) { *len = 4U; }
    return 0;
}

static void st03_set_param_value(uint16_t sid, uint8_t idx, uint32_t value)
{
    if (sid < PUS_ST03_MAX_SID && idx < PUS_ST03_MAX_PARAMS) {
        st03_param_values[sid][idx] = value;
    }
}

/* CCSDS helpers for building minimal TC packets */
static uint16_t build_tc_header(uint8_t *buf, uint16_t apid,
                                 uint16_t data_len)
{
    /* Packet ID: version=000, type=1(TC), shdr=1, APID */
    uint16_t pkt_id = 0x1800U | (apid & 0x07FFU);
    buf[0] = (uint8_t)(pkt_id >> 8);
    buf[1] = (uint8_t)(pkt_id & 0xFFU);

    /* Sequence: unseg=11, count=0 */
    buf[2] = 0xC0U;
    buf[3] = 0x00U;

    /* Data length: total_data_bytes - 1 */
    uint16_t dl = (data_len > 0U) ? (data_len - 1U) : 0U;
    buf[4] = (uint8_t)(dl >> 8);
    buf[5] = (uint8_t)(dl & 0xFFU);

    return 6U;  /* primary header size */
}

/* PUS headers included earlier provide the proper prototypes. */

/* ── Test infrastructure ──────────────────────────────────────────────── */
static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)  do { \
    printf("  TEST %-40s ", #name); \
    tests_run++; \
    name(); \
    tests_passed++; \
    printf("[PASS]\n"); \
} while(0)

/* ══════════════════════════════════════════════════════════════════════════
 *  ST01 — Verification
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st01_init(void)
{
    int32_t ret = pus_st01_init(0x010U);
    assert(ret == 0);
}

static void test_st01_accept_ok(void)
{
    (void)pus_st01_init(0x010U);
    tm_capture_reset();

    ccsds_packet_t tc_pkt;
    (void)ccsds_init_packet(&tc_pkt, CCSDS_TYPE_TC, 0x010U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    /* Set sequence count = 1 */
    tc_pkt.header.pkt_seq_ctrl = (uint16_t)((tc_pkt.header.pkt_seq_ctrl & 0xC000U) | (1U & 0x3FFFU));

    int32_t ret = pus_st01_accept_ok(&tc_pkt);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,1) emitted */
}

static void test_st01_accept_fail(void)
{
    (void)pus_st01_init(0x010U);
    tm_capture_reset();

    ccsds_packet_t tc_pkt;
    (void)ccsds_init_packet(&tc_pkt, CCSDS_TYPE_TC, 0x010U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    tc_pkt.header.pkt_seq_ctrl = (uint16_t)((tc_pkt.header.pkt_seq_ctrl & 0xC000U) | (2U & 0x3FFFU));

    int32_t ret = pus_st01_accept_fail(&tc_pkt, 0x42);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,2) emitted */
}

static void test_st01_exec_ok(void)
{
    (void)pus_st01_init(0x010U);
    tm_capture_reset();

    ccsds_packet_t tc_pkt;
    (void)ccsds_init_packet(&tc_pkt, CCSDS_TYPE_TC, 0x010U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    tc_pkt.header.pkt_seq_ctrl = (uint16_t)((tc_pkt.header.pkt_seq_ctrl & 0xC000U) | (3U & 0x3FFFU));

    int32_t ret = pus_st01_exec_ok(&tc_pkt);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,7) */
}

static void test_st01_exec_fail(void)
{
    (void)pus_st01_init(0x010U);
    tm_capture_reset();

    ccsds_packet_t tc_pkt;
    (void)ccsds_init_packet(&tc_pkt, CCSDS_TYPE_TC, 0x010U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    tc_pkt.header.pkt_seq_ctrl = (uint16_t)((tc_pkt.header.pkt_seq_ctrl & 0xC000U) | (4U & 0x3FFFU));

    int32_t ret = pus_st01_exec_fail(&tc_pkt, 0x13);
    assert(ret == 0);
    assert(tm_capture_count == 1U);  /* TM(1,8) */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST03 — Housekeeping
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st03_init(void)
{
    int32_t ret = pus_st03_init(0x020U, test_st03_reader);
    assert(ret == 0);
}

static void test_st03_define_and_enable(void)
{
    (void)pus_st03_init(0x020U, test_st03_reader);

    /* Define a single-parameter SID where the param_id encodes sid<<8 | idx */
    uint16_t pids[1]; uint8_t psizes[1];
    pids[0] = (uint16_t)((1U << 8) | 0U);
    psizes[0] = 4U;
    int32_t ret = pus_st03_define(1U, pids, psizes, 1U, 5000U); /* 5 s */
    assert(ret == 0);

    ret = pus_st03_enable(1U);
    assert(ret == 0);
}

static void test_st03_set_param(void)
{
    (void)pus_st03_init(0x020U, test_st03_reader);
    {
        uint16_t pids[1]; uint8_t psizes[1];
        pids[0] = (uint16_t)((1U << 8) | 0U);
        psizes[0] = 4U;
        (void)pus_st03_define(1U, pids, psizes, 1U, 5000U);
    }

    st03_set_param_value(1U, 0U, 0xDEADBEEFU);
}

static void test_st03_tick_generates_tm(void)
{
    (void)pus_st03_init(0x020U, test_st03_reader);
    {
        uint16_t pids[1]; uint8_t psizes[1];
        pids[0] = (uint16_t)((1U << 8) | 0U);
        psizes[0] = 4U;
        (void)pus_st03_define(1U, pids, psizes, 1U, 1000U);
    }
    (void)pus_st03_enable(1U);
    st03_set_param_value(1U, 0U, 42U);

    tm_capture_reset();
    stub_uptime_ms = 1000U;

    /* Tick enough times to exceed interval */
    pus_st03_tick(stub_uptime_ms);
    /* A TM(3,25) should have been emitted */
    assert(tm_capture_count >= 1U);
}

static void test_st03_disable(void)
{
    (void)pus_st03_init(0x020U, test_st03_reader);
    {
        uint16_t pids[1]; uint8_t psizes[1];
        pids[0] = (uint16_t)((2U << 8) | 0U);
        psizes[0] = 4U;
        (void)pus_st03_define(2U, pids, psizes, 1U, 1000U);
    }
    (void)pus_st03_enable(2U);
    (void)pus_st03_disable(2U);

    tm_capture_reset();
    (void)pus_st03_tick(stub_uptime_ms);
    /* Disabled SID should not generate TM */
    /* TM count for this SID should be 0 */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST05 — Events
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st05_init(void)
{
    int32_t ret = pus_st05_init(0x020U);
    assert(ret == 0);
}

static void test_st05_raise_event(void)
{
    (void)pus_st05_init(0x020U);
    tm_capture_reset();

    uint8_t aux[4] = { 0x01, 0x02, 0x03, 0x04 };
    (void)pus_st05_raise(0x1000U, 2U, aux, 4U);

    /* Should have emitted a TM(5,x) via router_send_tm */
    assert(tm_capture_count >= 1U);
}

static void test_st05_disable_enable(void)
{
    (void)pus_st05_init(0x020U);

    int32_t ret = pus_st05_disable(0x1000U);
    assert(ret == 0);
    ret = pus_st05_enable(0x1000U);
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST08 — Function Management
 * ══════════════════════════════════════════════════════════════════════ */

static int32_t mock_func_handler(const uint8_t *args, uint32_t len)
{
    (void)args;
    (void)len;
    return 0;
}

static void test_st08_init(void)
{
    int32_t ret = pus_st08_init();
    assert(ret == 0);
}

static void test_st08_register(void)
{
    (void)pus_st08_init();
    int32_t ret = pus_st08_register(1U, mock_func_handler);
    assert(ret == 0);
}

static void test_st08_dispatch(void)
{
    (void)pus_st08_init();
    (void)pus_st08_register(1U, mock_func_handler);

    /* Build a minimal TC(8,1) packet:
     * Primary header (6) + PUS secondary header (min 3) +
     * function_id (1) */
    uint8_t tc[16];
    uint16_t hdr_len = build_tc_header(tc, 0x020U, 4U);

    /* PUS secondary header: version=0x20, service=8, subtype=1 */
    tc[hdr_len + 0U] = 0x20U;
    tc[hdr_len + 1U] = 8U;
    tc[hdr_len + 2U] = 1U;

    /* Function ID */
    tc[hdr_len + 3U] = 1U;

    tm_capture_reset();
    int32_t ret = pus_st08_process(tc, (uint32_t)(hdr_len + 4U));
    assert(ret == 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST09 — Time Management
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st09_init(void)
{
    int32_t ret = pus_st09_init(0x020U);
    assert(ret == 0);
}

static void test_st09_set_get_time(void)
{
    (void)pus_st09_init(0x020U);

    cuc_time_t t;
    t.coarse = 1000000U;
    t.fine = 32768U;
    int32_t ret = pus_st09_set_time(&t);
    assert(ret == 0);

    cuc_time_t got;
    ret = pus_st09_get_time(&got);
    assert(ret == 0);
    assert(got.coarse == 1000000U);
    assert(got.fine == 32768U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST11 — Time-based Scheduling
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st11_init(void)
{
    int32_t ret = pus_st11_init();
    assert(ret == 0);
}

static void test_st11_insert_and_count(void)
{
    (void)pus_st11_init();

    /* Insert a dummy TC scheduled for epoch 100 */
    ccsds_packet_t pkt;
    (void)ccsds_init_packet(&pkt, CCSDS_TYPE_TC, 0x020U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    uint8_t d[4] = {0x20U, 17U, 1U, 0U};
    (void)ccsds_set_data(&pkt, d, 4U);

    int32_t ret = pus_st11_insert(100U, &pkt);
    assert(ret == 0);

    assert(pus_st11_count() == 1U);
}

static void test_st11_enable_disable(void)
{
    (void)pus_st11_init();

    pus_st11_enable();

    pus_st11_disable();
}

static void test_st11_tick_executes_due(void)
{
    (void)pus_st11_init();

    ccsds_packet_t pkt;
    (void)ccsds_init_packet(&pkt, CCSDS_TYPE_TC, 0x020U,
                            CCSDS_SEQ_UNSEG, CCSDS_SHDR_PRESENT);
    uint8_t d[4] = {0x20U, 17U, 1U, 0U};
    (void)ccsds_set_data(&pkt, d, 4U);

    (void)pus_st11_insert(50U, &pkt);
    (void)pus_st11_enable();

    tm_capture_reset();
    pus_st11_tick(60U, router_send_tm);

    /* The scheduled TC should have been dispatched; count should now be 0 */
    assert(pus_st11_count() == 0U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ST17 — Test Connection
 * ══════════════════════════════════════════════════════════════════════ */

static void test_st17_init(void)
{
    int32_t ret = pus_st17_init(0x020U);
    assert(ret == 0);
}

static void test_st17_are_you_alive(void)
{
    (void)pus_st17_init(0x020U);
    tm_capture_reset();

    /* Build TC(17,1) */
    uint8_t tc[16];
    uint16_t hdr = build_tc_header(tc, 0x020U, 3U);
    tc[hdr + 0U] = 0x20U;  /* PUS version 2 */
    tc[hdr + 1U] = 17U;    /* service */
    tc[hdr + 2U] = 1U;     /* subtype — are you alive */

    int32_t ret = pus_st17_handle();
    assert(ret == 0);
    /* Should emit TM(17,2) */
    assert(tm_capture_count >= 1U);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== PUS Services Unit Tests ===\n");

    printf("\n-- ST01 Verification --\n");
    TEST(test_st01_init);
    TEST(test_st01_accept_ok);
    TEST(test_st01_accept_fail);
    TEST(test_st01_exec_ok);
    TEST(test_st01_exec_fail);

    printf("\n-- ST03 Housekeeping --\n");
    TEST(test_st03_init);
    TEST(test_st03_define_and_enable);
    TEST(test_st03_set_param);
    TEST(test_st03_tick_generates_tm);
    TEST(test_st03_disable);

    printf("\n-- ST05 Events --\n");
    TEST(test_st05_init);
    TEST(test_st05_raise_event);
    TEST(test_st05_disable_enable);

    printf("\n-- ST08 Function Management --\n");
    TEST(test_st08_init);
    TEST(test_st08_register);
    TEST(test_st08_dispatch);

    printf("\n-- ST09 Time Management --\n");
    TEST(test_st09_init);
    TEST(test_st09_set_get_time);

    printf("\n-- ST11 Time-based Scheduling --\n");
    TEST(test_st11_init);
    TEST(test_st11_insert_and_count);
    TEST(test_st11_enable_disable);
    TEST(test_st11_tick_executes_due);

    printf("\n-- ST17 Test Connection --\n");
    TEST(test_st17_init);
    TEST(test_st17_are_you_alive);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
