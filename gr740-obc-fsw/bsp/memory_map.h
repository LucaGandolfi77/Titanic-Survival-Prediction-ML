/**
 * @file memory_map.h
 * @brief AMBA Plug-and-Play memory map and device enumeration for GR740.
 *
 * Provides structures and functions to scan the AMBA bus for devices
 * and retrieve their base addresses and IRQ assignments.
 *
 * @reference GR740 User Manual, GRLIB AMBA Plug-and-Play specification
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef MEMORY_MAP_H
#define MEMORY_MAP_H

#include <stdint.h>

/* ======================================================================
 * AMBA Plug-and-Play Configuration Area
 * ====================================================================== */
#define AMBA_PNP_AHB_BASE       0xFFFFF000U  /**< AHB P&P area base      */
#define AMBA_PNP_APB_BASE       0x800FF000U  /**< APB P&P area base      */
#define AMBA_PNP_ENTRY_SIZE     32U          /**< 8 words per entry       */
#define AMBA_PNP_MAX_AHB        64U          /**< Max AHB masters/slaves  */
#define AMBA_PNP_MAX_APB        512U         /**< Max APB slaves          */

/* ======================================================================
 * GRLIB Vendor and Device IDs
 * ====================================================================== */
#define VENDOR_GAISLER           0x01U

/* AHB device IDs */
#define GAISLER_LEON4            0x048U
#define GAISLER_GRSPW2           0x029U
#define GAISLER_GRCAN            0x034U
#define GAISLER_AHB2AHB          0x020U

/* APB device IDs */
#define GAISLER_APBUART          0x00CU
#define GAISLER_IRQAMP           0x00DU
#define GAISLER_GPTIMER          0x011U
#define GAISLER_SPICTRL          0x02DU
#define GAISLER_I2CMST           0x028U
#define GAISLER_GRGPIO           0x01AU

/* ======================================================================
 * AMBA Device Info Structure
 * ====================================================================== */

/**
 * @brief Information about a discovered AMBA device.
 */
typedef struct {
    uint16_t vendor;        /**< Vendor ID                               */
    uint16_t device;        /**< Device ID                               */
    uint32_t base_addr;     /**< Base address (from BAR)                 */
    uint32_t irq;           /**< IRQ number                              */
    uint8_t  version;       /**< Device version                          */
    uint8_t  index;         /**< Instance index (0, 1, 2...)             */
} amba_dev_info_t;

/* ======================================================================
 * Function Prototypes
 * ====================================================================== */

/**
 * @brief Scan the AMBA AHB bus for devices.
 * @param[out] devs Array to store discovered devices.
 * @param[in]  max_devs Maximum entries in devs array.
 * @return Number of devices found (>=0), or negative on error.
 */
int32_t amba_scan_ahb(amba_dev_info_t *devs, uint32_t max_devs);

/**
 * @brief Scan the AMBA APB bus for devices.
 * @param[out] devs Array to store discovered devices.
 * @param[in]  max_devs Maximum entries in devs array.
 * @return Number of devices found (>=0), or negative on error.
 */
int32_t amba_scan_apb(amba_dev_info_t *devs, uint32_t max_devs);

/**
 * @brief Find a specific device on the APB bus.
 * @param[in]  vendor Vendor ID to search for.
 * @param[in]  device Device ID to search for.
 * @param[in]  index  Instance index (0 for first instance).
 * @param[out] info   Pointer to store device info.
 * @return 0 on success, -1 if not found.
 */
int32_t amba_find_apb(uint16_t vendor, uint16_t device,
                      uint8_t index, amba_dev_info_t *info);

/**
 * @brief Find a specific device on the AHB bus.
 * @param[in]  vendor Vendor ID to search for.
 * @param[in]  device Device ID to search for.
 * @param[in]  index  Instance index (0 for first instance).
 * @param[out] info   Pointer to store device info.
 * @return 0 on success, -1 if not found.
 */
int32_t amba_find_ahb(uint16_t vendor, uint16_t device,
                      uint8_t index, amba_dev_info_t *info);

#endif /* MEMORY_MAP_H */
