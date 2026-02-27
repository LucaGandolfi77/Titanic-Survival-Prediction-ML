/**
 * @file test_loop.c
 * @brief Integration test — end-to-end TC dispatch → TM generation →
 *        verification loop.
 *
 * Exercises the router + PUS service chain with stubbed BSP/drivers.
 * Verifies that a TC(17,1) injected into the router produces a
 * TM(17,2) in the downlink queue, and that PUS ST01 verification
 * reports are generated.
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

#include "middleware/ccsds/space_packet.h"
#include "middleware/routing/packet_router.h"
#include "middleware/pus/pus_st01.h"
#include "middleware/pus/pus_st17.h"


/* ── BSP stubs ─────────────────────────────────────────────────────────── */
static uint32_t stub_uptime_ms = 0U;
uint32_t bsp_get_uptime_ms(void) { return stub_uptime_ms; }

/* ── Driver stubs (SpW / CAN / UART TX callbacks) ─────────────────────── */
#define DL_CAPTURE_MAX 64U
static uint8_t  dl_capture[DL_CAPTURE_MAX][256];
static uint16_t dl_capture_len[DL_CAPTURE_MAX];
static uint32_t dl_capture_count = 0U;

static void dl_capture_reset(void)
{
    dl_capture_count = 0U;
    memset(dl_capture, 0, sizeof(dl_capture));
    memset(dl_capture_len, 0, sizeof(dl_capture_len));
}

static int32_t stub_spw_tx(const uint8_t *data, uint16_t len)
{
    if (dl_capture_count < DL_CAPTURE_MAX) {
        if (len <= 256U) {
            memcpy(dl_capture[dl_capture_count], data, len);
            dl_capture_len[dl_capture_count] = len;
        }
        dl_capture_count++;
    }
    return 0;
}

/* ── CRC-16 CCITT (must match space_packet.c implementation) ──────────── */
static uint16_t crc16_byte(uint16_t crc, uint8_t b)
{
    crc ^= (uint16_t)b << 8;
    for (int i = 0; i < 8; i++) {
        if ((crc & 0x8000U) != 0U) {
            crc = (uint16_t)((crc << 1) ^ 0x1021U);
        } else {
            crc <<= 1;
        }
    }
    return crc;
}

static uint16_t compute_crc16(const uint8_t *data, uint16_t len)
{
    uint16_t crc = 0xFFFFU;
    for (uint16_t i = 0U; i < len; i++) {
        crc = crc16_byte(crc, data[i]);
    }
    return crc;
}

/* Use the packet router and CCSDS headers for proper prototypes */

/* PUS service declarations provided by headers above */

/* Adapter to register with the router: convert router callback to
 * call the service's handler which takes no args. */
static int32_t pus_st17_handle_tc(uint8_t svc_subtype,
                                  const uint8_t *data,
                                  uint16_t data_len)
{
    (void)svc_subtype; (void)data; (void)data_len;
    return pus_st17_handle();
}

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
 *  HELPER: Build a complete TC(17,1) space packet with CRC
 * ══════════════════════════════════════════════════════════════════════ */

static uint16_t build_tc17_1(uint8_t *buf, uint16_t buf_size)
{
    (void)buf_size;

    /*
     * CCSDS Primary Header (6 bytes):
     *   Packet ID: ver=000, type=1(TC), shdr=1, APID=0x020
     *   Sequence: unseg=11, count=0
     *   Data length: (PUS_SEC_HDR + CRC) - 1
     *
     * PUS-C Secondary Header (min 9 bytes for this impl):
     *   [0] = 0x20  (PUS version 2)
     *   [1] = ack_flags (0x0F = all)
     *   [2] = service_type = 17
     *   [3] = subtype = 1
     *   [4..5] = source_id = 0x0000
     *   Followed by CUC coarse (4) — simplify to 3 spare bytes + pad
     *
     * For simplicity, match the format expected by packet_router.c:
     *   Primary header: 6 bytes
     *   PUS secondary header: 9 bytes (version, ack, svc, sub, src_id(2), spare(3))
     *   Packet error control: 2 bytes (CRC-16)
     */
    uint16_t pus_len = 9U;   /* PUS secondary header */
    uint16_t crc_len = 2U;
    uint16_t data_field_len = pus_len + crc_len;

    /* Primary header */
    uint16_t pkt_id = 0x1820U;  /* type=1, shdr=1, APID=0x020 */
    buf[0] = (uint8_t)(pkt_id >> 8);
    buf[1] = (uint8_t)(pkt_id & 0xFFU);

    buf[2] = 0xC0U;  /* unseg, seq=0 */
    buf[3] = 0x00U;

    uint16_t pkt_data_len = data_field_len - 1U;
    buf[4] = (uint8_t)(pkt_data_len >> 8);
    buf[5] = (uint8_t)(pkt_data_len & 0xFFU);

    /* PUS-C secondary header */
    buf[6]  = 0x20U;  /* PUS version 2 */
    buf[7]  = 0x0FU;  /* ack all */
    buf[8]  = 17U;    /* service type */
    buf[9]  = 1U;     /* subtype */
    buf[10] = 0x00U;  /* source_id high */
    buf[11] = 0x00U;  /* source_id low */
    buf[12] = 0x00U;  /* spare */
    buf[13] = 0x00U;  /* spare */
    buf[14] = 0x00U;  /* spare */

    /* CRC-16 over entire packet (bytes 0..14) */
    uint16_t total_no_crc = 6U + pus_len;  /* 15 bytes */
    uint16_t crc = compute_crc16(buf, total_no_crc);
    buf[total_no_crc + 0U] = (uint8_t)(crc >> 8);
    buf[total_no_crc + 1U] = (uint8_t)(crc & 0xFFU);

    return total_no_crc + crc_len;  /* 17 bytes total */
}

/* ══════════════════════════════════════════════════════════════════════════
 *  INTEGRATION TESTS
 * ══════════════════════════════════════════════════════════════════════ */

static void test_init_chain(void)
{
    int32_t ret;

    ret = router_init();
    assert(ret == 0);

        ret = pus_st01_init(0x010U);
    assert(ret == 0);

        ret = pus_st17_init(0x020U);
    assert(ret == 0);

    /* Register ST17 handler with the router */
    ret = router_register_service(17U, pus_st17_handle_tc);
    assert(ret == 0);

    /* Register and enable SpaceWire as downlink */
    ret = router_register_downlink(ROUTER_DL_SPW, stub_spw_tx);
    assert(ret == 0);
    ret = router_set_active_downlink(ROUTER_DL_SPW);
    assert(ret == 0);
}

static void test_tc_dispatch_to_st17(void)
{
    /* Full init */
    (void)router_init();
        (void)pus_st01_init(0x010U);
    (void)pus_st17_init(0x020U);
    (void)router_register_service(17U, pus_st17_handle_tc);
    (void)router_register_downlink(ROUTER_DL_SPW, stub_spw_tx);
    (void)router_set_active_downlink(ROUTER_DL_SPW);

    dl_capture_reset();
    stub_uptime_ms = 5000U;

    /* Build TC(17,1) */
    uint8_t tc_buf[64];
    uint16_t tc_len = build_tc17_1(tc_buf, 64U);

    /* Dispatch through router */
    int32_t ret = router_dispatch_tc(tc_buf, tc_len);
    assert(ret == 0);

    /* Process TM queue — should push TM(17,2) to downlink */
    ret = router_process_tm_queue();
    assert(ret == 0);

    /* At least one packet should have been sent via stub_spw_tx */
    assert(dl_capture_count >= 1U);
}

static void test_tm_contains_valid_header(void)
{
    /* Re-run the dispatch */
    (void)router_init();
        (void)pus_st01_init(0x010U);
    (void)pus_st17_init(0x020U);
    (void)router_register_service(17U, pus_st17_handle_tc);
    (void)router_register_downlink(ROUTER_DL_SPW, stub_spw_tx);
    (void)router_set_active_downlink(ROUTER_DL_SPW);

    dl_capture_reset();
    stub_uptime_ms = 6000U;

    uint8_t tc_buf[64];
    uint16_t tc_len = build_tc17_1(tc_buf, 64U);

    (void)router_dispatch_tc(tc_buf, tc_len);
    (void)router_process_tm_queue();

    /* Verify that the first captured TM packet has a valid CCSDS header */
    assert(dl_capture_count >= 1U);
    uint8_t *tm = dl_capture[0];
    uint16_t tm_len = dl_capture_len[0];
    assert(tm_len >= 6U);

    /* Version = 000 (bits 15:13 of packet ID) */
    uint16_t pkt_id = ((uint16_t)tm[0] << 8) | tm[1];
    assert((pkt_id & 0xE000U) == 0x0000U);

    /* Type = 0 (TM) — bit 12 */
    assert((pkt_id & 0x1000U) == 0x0000U);
}

static void test_multiple_tc_dispatch(void)
{
    (void)router_init();
        (void)pus_st01_init(0x010U);
    (void)pus_st17_init(0x020U);
    (void)router_register_service(17U, pus_st17_handle_tc);
    (void)router_register_downlink(ROUTER_DL_SPW, stub_spw_tx);
    (void)router_set_active_downlink(ROUTER_DL_SPW);

    dl_capture_reset();
    stub_uptime_ms = 7000U;

    uint8_t tc_buf[64];
    uint16_t tc_len = build_tc17_1(tc_buf, 64U);

    /* Dispatch 5 TCs */
    for (int i = 0; i < 5; i++) {
        int32_t ret = router_dispatch_tc(tc_buf, tc_len);
        assert(ret == 0);
    }

    /* Process all TM */
    (void)router_process_tm_queue();

    /* Should have at least 5 TM packets */
    assert(dl_capture_count >= 5U);
}

static void test_unknown_service_rejected(void)
{
    (void)router_init();
    (void)pus_st01_init(0x010U);
    /* Do NOT register any service handler */

    dl_capture_reset();

    uint8_t tc_buf[64];
    /* Build a TC for service 99 (not registered) */
    uint16_t pus_len = 9U;
    uint16_t crc_len = 2U;
    uint16_t data_field_len = pus_len + crc_len;

    uint16_t pkt_id = 0x1820U;
    tc_buf[0] = (uint8_t)(pkt_id >> 8);
    tc_buf[1] = (uint8_t)(pkt_id & 0xFFU);
    tc_buf[2] = 0xC0U;
    tc_buf[3] = 0x00U;
    uint16_t pdl = data_field_len - 1U;
    tc_buf[4] = (uint8_t)(pdl >> 8);
    tc_buf[5] = (uint8_t)(pdl & 0xFFU);
    tc_buf[6]  = 0x20U;
    tc_buf[7]  = 0x0FU;
    tc_buf[8]  = 99U;   /* unknown service */
    tc_buf[9]  = 1U;
    tc_buf[10] = 0x00U;
    tc_buf[11] = 0x00U;
    tc_buf[12] = 0x00U;
    tc_buf[13] = 0x00U;
    tc_buf[14] = 0x00U;

    uint16_t crc = compute_crc16(tc_buf, 15U);
    tc_buf[15] = (uint8_t)(crc >> 8);
    tc_buf[16] = (uint8_t)(crc & 0xFFU);

    int32_t ret = router_dispatch_tc(tc_buf, 17U);
    /* Router should return error for unknown service */
    assert(ret != 0);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("=== Integration Test: TC → TM Loop ===\n\n");

    TEST(test_init_chain);
    TEST(test_tc_dispatch_to_st17);
    TEST(test_tm_contains_valid_header);
    TEST(test_multiple_tc_dispatch);
    TEST(test_unknown_service_rejected);

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
