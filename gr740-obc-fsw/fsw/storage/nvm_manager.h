/**
 * @file nvm_manager.h
 * @brief Non-Volatile Memory Manager — EEPROM / MRAM abstraction.
 *
 * Provides sector-based read/write with CRC-16 integrity checking
 * and wear-leveling rotation across MRAM sectors.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef NVM_MANAGER_H
#define NVM_MANAGER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ──────────────────────────────────────────────────────── */
#define NVM_OK             0
#define NVM_ERR_PARAM     -1
#define NVM_ERR_CRC       -2
#define NVM_ERR_WRITE     -3
#define NVM_ERR_ERASE     -4
#define NVM_ERR_RANGE     -5

/* ── Configuration ─────────────────────────────────────────────────────── */
#define NVM_MRAM_BASE       0x20000000U     /**< MRAM base address       */
#define NVM_MRAM_SIZE       (8U * 1024U * 1024U)   /**< 8 MB            */
#define NVM_EEPROM_BASE     0x30000000U     /**< EEPROM base address     */
#define NVM_EEPROM_SIZE     (4U * 1024U * 1024U)   /**< 4 MB            */

#define NVM_SECTOR_SIZE     4096U           /**< 4 KB sector size        */
#define NVM_MAX_REGIONS     16U             /**< Max named regions       */
#define NVM_HDR_MAGIC       0x4E564D31U     /**< "NVM1" magic            */

/* ── NVM region descriptor ─────────────────────────────────────────────── */
typedef enum {
    NVM_STORE_MRAM   = 0,
    NVM_STORE_EEPROM = 1
} nvm_store_t;

/* ── Public API ────────────────────────────────────────────────────────── */

/**
 * @brief Initialise the NVM manager.
 * @return NVM_OK on success.
 */
int32_t nvm_init(void);

/**
 * @brief Register a named region.
 *
 * @param[in] region_id   Unique region ID (0..NVM_MAX_REGIONS-1).
 * @param[in] store       MRAM or EEPROM.
 * @param[in] offset      Byte offset within the store.
 * @param[in] size        Region size in bytes.
 * @return NVM_OK on success.
 */
int32_t nvm_register_region(uint8_t region_id, nvm_store_t store,
                             uint32_t offset, uint32_t size);

/**
 * @brief Write data to a region with CRC-16 header.
 *
 * @param[in] region_id  Region ID.
 * @param[in] data       Data buffer.
 * @param[in] len        Data length in bytes.
 * @return NVM_OK on success.
 */
int32_t nvm_write(uint8_t region_id, const uint8_t *data, uint32_t len);

/**
 * @brief Read data from a region, verifying CRC-16.
 *
 * @param[in]  region_id  Region ID.
 * @param[out] data       Output buffer.
 * @param[in]  max_len    Buffer capacity.
 * @param[out] actual_len Actual data length read.
 * @return NVM_OK on success, NVM_ERR_CRC on integrity fail.
 */
int32_t nvm_read(uint8_t region_id, uint8_t *data, uint32_t max_len,
                  uint32_t *actual_len);

/**
 * @brief Erase a region (fill with 0xFF).
 *
 * @param[in] region_id  Region ID.
 * @return NVM_OK on success.
 */
int32_t nvm_erase(uint8_t region_id);

/**
 * @brief Get CRC-16 of a memory range.
 *
 * @param[in] data  Data buffer.
 * @param[in] len   Length in bytes.
 * @return CRC-16 value.
 */
uint16_t nvm_crc16(const uint8_t *data, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif /* NVM_MANAGER_H */
