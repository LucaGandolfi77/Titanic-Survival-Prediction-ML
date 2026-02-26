/**
 * @file nvm_manager.c
 * @brief Non-Volatile Memory Manager implementation.
 *
 * Each region write prepends a 12-byte header:
 *   [4] magic  [4] length  [2] seq_num  [2] CRC-16(header + data)
 *
 * Reads verify CRC-16 and report integrity failures.
 * MRAM is byte-addressable; EEPROM writes use SPI interface.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "nvm_manager.h"

/* ── CRC-16 CCITT (same polynomial as CCSDS) ──────────────────────────── */
extern uint16_t ccsds_crc16(const uint8_t *data, uint16_t len);

/* ── Region descriptor ──────────────────────────────────────────────────── */
typedef struct {
    nvm_store_t store;
    uint32_t    base_addr;
    uint32_t    offset;
    uint32_t    size;
    uint16_t    seq_num;
    uint8_t     registered;
} nvm_region_t;

/* ── NVM header structure ──────────────────────────────────────────────── */
#define NVM_HDR_SIZE  12U

/* ── Module state ──────────────────────────────────────────────────────── */
static nvm_region_t regions[NVM_MAX_REGIONS];
static uint8_t      nvm_init_done = 0U;

/* ── Private helpers ───────────────────────────────────────────────────── */

static uint32_t region_addr(const nvm_region_t *r)
{
    return r->base_addr + r->offset;
}

static void mem_write_byte(uint32_t addr, uint8_t val)
{
    volatile uint8_t *p = (volatile uint8_t *)(uintptr_t)addr;
    *p = val;
}

static uint8_t mem_read_byte(uint32_t addr)
{
    volatile uint8_t *p = (volatile uint8_t *)(uintptr_t)addr;
    return *p;
}

static void mem_write_u32(uint32_t addr, uint32_t val)
{
    mem_write_byte(addr + 0U, (uint8_t)((val >> 24U) & 0xFFU));
    mem_write_byte(addr + 1U, (uint8_t)((val >> 16U) & 0xFFU));
    mem_write_byte(addr + 2U, (uint8_t)((val >> 8U)  & 0xFFU));
    mem_write_byte(addr + 3U, (uint8_t)(val & 0xFFU));
}

static uint32_t mem_read_u32(uint32_t addr)
{
    uint32_t v = 0U;
    v |= ((uint32_t)mem_read_byte(addr + 0U) << 24U);
    v |= ((uint32_t)mem_read_byte(addr + 1U) << 16U);
    v |= ((uint32_t)mem_read_byte(addr + 2U) << 8U);
    v |= ((uint32_t)mem_read_byte(addr + 3U));
    return v;
}

static void mem_write_u16(uint32_t addr, uint16_t val)
{
    mem_write_byte(addr + 0U, (uint8_t)((val >> 8U) & 0xFFU));
    mem_write_byte(addr + 1U, (uint8_t)(val & 0xFFU));
}

static uint16_t mem_read_u16(uint32_t addr)
{
    uint16_t v = 0U;
    v |= ((uint16_t)mem_read_byte(addr + 0U) << 8U);
    v |= ((uint16_t)mem_read_byte(addr + 1U));
    return v;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t nvm_init(void)
{
    uint32_t i;

    for (i = 0U; i < NVM_MAX_REGIONS; i++) {
        regions[i].registered = 0U;
        regions[i].seq_num    = 0U;
    }
    nvm_init_done = 1U;
    return NVM_OK;
}

int32_t nvm_register_region(uint8_t region_id, nvm_store_t store,
                             uint32_t offset, uint32_t size)
{
    nvm_region_t *r;

    if (region_id >= NVM_MAX_REGIONS) {
        return NVM_ERR_PARAM;
    }
    if (size == 0U) {
        return NVM_ERR_PARAM;
    }

    r = &regions[region_id];
    r->store     = store;
    r->offset    = offset;
    r->size      = size;
    r->seq_num   = 0U;
    r->registered = 1U;

    if (store == NVM_STORE_MRAM) {
        r->base_addr = NVM_MRAM_BASE;
    } else {
        r->base_addr = NVM_EEPROM_BASE;
    }

    return NVM_OK;
}

int32_t nvm_write(uint8_t region_id, const uint8_t *data, uint32_t len)
{
    nvm_region_t *r;
    uint32_t      addr;
    uint32_t      total;
    uint32_t      i;
    uint16_t      crc;
    uint8_t       hdr_buf[NVM_HDR_SIZE];

    if (region_id >= NVM_MAX_REGIONS) {
        return NVM_ERR_PARAM;
    }
    r = &regions[region_id];
    if (r->registered == 0U) {
        return NVM_ERR_PARAM;
    }
    if (data == (const uint8_t *)0) {
        return NVM_ERR_PARAM;
    }

    total = NVM_HDR_SIZE + len;
    if (total > r->size) {
        return NVM_ERR_RANGE;
    }

    addr = region_addr(r);

    /* Build header in temp buffer for CRC calculation */
    hdr_buf[0]  = (uint8_t)((NVM_HDR_MAGIC >> 24U) & 0xFFU);
    hdr_buf[1]  = (uint8_t)((NVM_HDR_MAGIC >> 16U) & 0xFFU);
    hdr_buf[2]  = (uint8_t)((NVM_HDR_MAGIC >> 8U)  & 0xFFU);
    hdr_buf[3]  = (uint8_t)(NVM_HDR_MAGIC & 0xFFU);
    hdr_buf[4]  = (uint8_t)((len >> 24U) & 0xFFU);
    hdr_buf[5]  = (uint8_t)((len >> 16U) & 0xFFU);
    hdr_buf[6]  = (uint8_t)((len >> 8U)  & 0xFFU);
    hdr_buf[7]  = (uint8_t)(len & 0xFFU);
    hdr_buf[8]  = (uint8_t)((r->seq_num >> 8U) & 0xFFU);
    hdr_buf[9]  = (uint8_t)(r->seq_num & 0xFFU);
    hdr_buf[10] = 0x00U;  /* CRC placeholder */
    hdr_buf[11] = 0x00U;

    /* Compute CRC over header (10 bytes) + data */
    /* First pass CRC: header bytes 0..9 */
    crc = nvm_crc16(hdr_buf, 10U);

    /* Continue CRC over data — manual accumulation */
    {
        /* Simple approach: copy approach for CRC */
        /* Re-use ccsds_crc16 which takes a contiguous buffer.
         * For efficiency, compute manually via polynomial.    */
        uint16_t poly = 0x1021U;
        uint16_t c    = crc;
        uint32_t j;
        uint8_t  bit;

        for (j = 0U; j < len; j++) {
            c ^= ((uint16_t)data[j] << 8U);
            for (bit = 0U; bit < 8U; bit++) {
                if ((c & 0x8000U) != 0U) {
                    c = (uint16_t)((c << 1U) ^ poly);
                } else {
                    c = (uint16_t)(c << 1U);
                }
            }
        }
        crc = c;
    }

    /* Write header */
    mem_write_u32(addr + 0U, NVM_HDR_MAGIC);
    mem_write_u32(addr + 4U, len);
    mem_write_u16(addr + 8U, r->seq_num);
    mem_write_u16(addr + 10U, crc);

    /* Write data */
    for (i = 0U; i < len; i++) {
        mem_write_byte(addr + NVM_HDR_SIZE + i, data[i]);
    }

    r->seq_num++;
    return NVM_OK;
}

int32_t nvm_read(uint8_t region_id, uint8_t *data, uint32_t max_len,
                  uint32_t *actual_len)
{
    nvm_region_t *r;
    uint32_t      addr;
    uint32_t      magic;
    uint32_t      stored_len;
    uint16_t      stored_crc;
    uint16_t      calc_crc;
    uint32_t      i;
    uint8_t       hdr_buf[10];

    if (region_id >= NVM_MAX_REGIONS) {
        return NVM_ERR_PARAM;
    }
    r = &regions[region_id];
    if (r->registered == 0U) {
        return NVM_ERR_PARAM;
    }
    if ((data == (uint8_t *)0) || (actual_len == (uint32_t *)0)) {
        return NVM_ERR_PARAM;
    }

    addr = region_addr(r);

    /* Read and verify magic */
    magic = mem_read_u32(addr + 0U);
    if (magic != NVM_HDR_MAGIC) {
        *actual_len = 0U;
        return NVM_ERR_CRC;
    }

    stored_len = mem_read_u32(addr + 4U);
    stored_crc = mem_read_u16(addr + 10U);

    if (stored_len > max_len) {
        *actual_len = 0U;
        return NVM_ERR_RANGE;
    }

    /* Rebuild header bytes for CRC */
    for (i = 0U; i < 10U; i++) {
        hdr_buf[i] = mem_read_byte(addr + i);
    }

    calc_crc = nvm_crc16(hdr_buf, 10U);

    /* Continue CRC over stored data */
    {
        uint16_t poly = 0x1021U;
        uint16_t c    = calc_crc;
        uint32_t j;
        uint8_t  bit;

        for (j = 0U; j < stored_len; j++) {
            uint8_t b = mem_read_byte(addr + NVM_HDR_SIZE + j);
            data[j] = b;
            c ^= ((uint16_t)b << 8U);
            for (bit = 0U; bit < 8U; bit++) {
                if ((c & 0x8000U) != 0U) {
                    c = (uint16_t)((c << 1U) ^ poly);
                } else {
                    c = (uint16_t)(c << 1U);
                }
            }
        }
        calc_crc = c;
    }

    if (calc_crc != stored_crc) {
        *actual_len = 0U;
        return NVM_ERR_CRC;
    }

    *actual_len = stored_len;
    return NVM_OK;
}

int32_t nvm_erase(uint8_t region_id)
{
    nvm_region_t *r;
    uint32_t      addr;
    uint32_t      i;

    if (region_id >= NVM_MAX_REGIONS) {
        return NVM_ERR_PARAM;
    }
    r = &regions[region_id];
    if (r->registered == 0U) {
        return NVM_ERR_PARAM;
    }

    addr = region_addr(r);
    for (i = 0U; i < r->size; i++) {
        mem_write_byte(addr + i, 0xFFU);
    }

    return NVM_OK;
}

uint16_t nvm_crc16(const uint8_t *data, uint32_t len)
{
    uint16_t crc  = 0xFFFFU;
    uint16_t poly = 0x1021U;
    uint32_t i;
    uint8_t  bit;

    for (i = 0U; i < len; i++) {
        crc ^= ((uint16_t)data[i] << 8U);
        for (bit = 0U; bit < 8U; bit++) {
            if ((crc & 0x8000U) != 0U) {
                crc = (uint16_t)((crc << 1U) ^ poly);
            } else {
                crc = (uint16_t)(crc << 1U);
            }
        }
    }
    return crc;
}
