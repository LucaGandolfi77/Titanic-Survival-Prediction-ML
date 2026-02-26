/**
 * @file gr740_init.h
 * @brief GR740 Board Support Package initialization interface.
 *
 * Provides low-level hardware initialization for the GR740 LEON4FT
 * processor including clock setup, memory controller configuration,
 * and basic peripheral initialization.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef GR740_INIT_H
#define GR740_INIT_H

#include <stdint.h>

/* ======================================================================
 * Return Codes
 * ====================================================================== */
#define BSP_OK                  0       /**< Success                     */
#define BSP_ERR_CLK             (-1)    /**< Clock init failure          */
#define BSP_ERR_MEM             (-2)    /**< Memory init failure         */
#define BSP_ERR_UART            (-3)    /**< UART init failure           */
#define BSP_ERR_IRQ             (-4)    /**< IRQ controller init failure */
#define BSP_ERR_NVM             (-5)    /**< NVM init failure            */

/* ======================================================================
 * Function Prototypes
 * ====================================================================== */

/**
 * @brief Full BSP initialization sequence.
 *
 * Initializes the GR740 hardware in the following order:
 * 1. Configure PLL to 50 MHz system clock
 * 2. Setup memory controller (SRAM with ECC)
 * 3. Initialize UART0 for debug output (115200 baud)
 * 4. Configure IRQAMP interrupt controller
 * 5. Initialize MRAM via SPI for parameter restore
 * 6. Start system timer (1 ms tick)
 *
 * @return BSP_OK on success, negative error code on failure.
 */
int32_t bsp_init(void);

/**
 * @brief Get system uptime in milliseconds.
 *
 * Returns elapsed time since bsp_init() was called, based on
 * the GPTIMER system tick counter.
 *
 * @return Uptime in milliseconds (wraps at ~49.7 days).
 */
uint32_t bsp_get_uptime_ms(void);

/**
 * @brief Get system uptime in seconds.
 * @return Uptime in seconds.
 */
uint32_t bsp_get_uptime_s(void);

/**
 * @brief Perform a controlled system reset.
 *
 * Saves critical state to MRAM before resetting the processor.
 */
void bsp_system_reset(void) __attribute__((noreturn));

/**
 * @brief Enable global interrupts.
 */
void bsp_enable_interrupts(void);

/**
 * @brief Disable global interrupts.
 * @return Previous interrupt state (for restore).
 */
uint32_t bsp_disable_interrupts(void);

/**
 * @brief Restore interrupt state.
 * @param[in] state Previous state from bsp_disable_interrupts().
 */
void bsp_restore_interrupts(uint32_t state);

/**
 * @brief Early console print (before full UART driver init).
 *
 * Polled-mode output on UART0. Safe to call at any init stage.
 *
 * @param[in] str Null-terminated string to print.
 */
void bsp_early_print(const char *str);

/**
 * @brief Read OBC board temperature from sensor.
 * @return Temperature in degrees Celsius, or INT32_MIN on error.
 */
int32_t bsp_read_temperature(void);

/**
 * @brief Enable ECC scrubbing on SRAM.
 * @return BSP_OK on success, negative on failure.
 */
int32_t bsp_enable_ecc_scrub(void);

#endif /* GR740_INIT_H */
