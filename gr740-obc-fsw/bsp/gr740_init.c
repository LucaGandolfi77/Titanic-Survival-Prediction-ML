/**
 * @file gr740_init.c
 * @brief GR740 Board Support Package — Low-level hardware initialization.
 *
 * Implements clock configuration, memory controller setup, UART0 debug
 * console, IRQAMP interrupt controller initialization, SRAM ECC enablement,
 * and MRAM initialization for parameter restore.
 *
 * @reference GR740 User Manual (GR740-UM), GRLIB IP Core User Manual
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "gr740_init.h"
#include "irq_handler.h"
#include "memory_map.h"
#include "../config/hw_config.h"
#include "../config/mission_config.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* ======================================================================
 * Private Constants
 * ====================================================================== */

/** LEON4 ASI for cache/memory control */
#define ASI_LEON_CACHEREGS      0x02U
#define ASI_LEON_BYPASS         0x1CU

/** Memory controller configuration register (SDRAM/SRAM) */
#define MEMCTRL_BASE            0x80000000U
#define MEMCTRL_MCFG1           0x00U   /**< Memory config register 1    */
#define MEMCTRL_MCFG2           0x04U   /**< Memory config register 2    */
#define MEMCTRL_MCFG3           0x08U   /**< Memory config register 3    */

/** SRAM ECC control — LEON4 FTAHBRAM */
#define FTAHBRAM_BASE           0x80001000U
#define FTAHBRAM_STATUS         0x00U   /**< Status/control register     */
#define FTAHBRAM_ECC_SCRUB      (1U << 0)   /**< Enable ECC scrubbing   */
#define FTAHBRAM_ECC_CE_IRQ     (1U << 1)   /**< Correctable error IRQ  */
#define FTAHBRAM_ECC_UE_IRQ     (1U << 2)   /**< Uncorrectable error IRQ*/

/* ======================================================================
 * Static Variables
 * ====================================================================== */

/** System uptime counter (milliseconds) — updated by timer ISR */
static volatile uint32_t s_uptime_ms = 0U;

/** BSP initialization complete flag */
static volatile uint8_t s_bsp_initialized = 0U;

/* ======================================================================
 * Forward Declarations (private functions)
 * ====================================================================== */

static int32_t bsp_init_clock(void);
static int32_t bsp_init_memory(void);
static int32_t bsp_init_uart0(void);
static int32_t bsp_init_irqamp(void);
static int32_t bsp_init_systick(void);
static int32_t bsp_init_mram(void);
static void    bsp_systick_isr(void);

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Full BSP initialization sequence.
 * @return BSP_OK on success, negative error code on failure.
 */
int32_t bsp_init(void)
{
    int32_t rc;

    /* Step 1: Configure system clock to 50 MHz */
    rc = bsp_init_clock();
    if (rc != BSP_OK) {
        return BSP_ERR_CLK;
    }

    /* Step 2: Configure memory controller (SRAM with ECC) */
    rc = bsp_init_memory();
    if (rc != BSP_OK) {
        return BSP_ERR_MEM;
    }

    /* Step 3: Initialize UART0 for debug output */
    rc = bsp_init_uart0();
    if (rc != BSP_OK) {
        return BSP_ERR_UART;
    }

    bsp_early_print("[BSP] UART0 initialized at 115200 baud\r\n");

    /* Step 4: Configure IRQAMP interrupt controller */
    rc = bsp_init_irqamp();
    if (rc != BSP_OK) {
        bsp_early_print("[BSP] ERROR: IRQAMP init failed\r\n");
        return BSP_ERR_IRQ;
    }
    bsp_early_print("[BSP] IRQAMP interrupt controller ready\r\n");

    /* Step 5: Enable ECC scrubbing on SRAM */
    rc = bsp_enable_ecc_scrub();
    if (rc != BSP_OK) {
        bsp_early_print("[BSP] WARNING: ECC scrub enable failed\r\n");
        /* Non-fatal, continue */
    } else {
        bsp_early_print("[BSP] SRAM ECC scrubbing enabled\r\n");
    }

    /* Step 6: Initialize MRAM via SPI for parameter restore */
    rc = bsp_init_mram();
    if (rc != BSP_OK) {
        bsp_early_print("[BSP] WARNING: MRAM init failed\r\n");
        /* Non-fatal, continue with defaults */
    } else {
        bsp_early_print("[BSP] MRAM initialized for NVM access\r\n");
    }

    /* Step 7: Start system tick timer (1 ms period) */
    rc = bsp_init_systick();
    if (rc != BSP_OK) {
        bsp_early_print("[BSP] ERROR: System timer init failed\r\n");
        return BSP_ERR_CLK;
    }
    bsp_early_print("[BSP] System tick timer started (1 ms)\r\n");

    s_bsp_initialized = 1U;

    bsp_early_print("[BSP] GR740 OBC initialization complete\r\n");
    bsp_early_print("[BSP] FSW Version: " FSW_VERSION_STRING "\r\n");

    return BSP_OK;
}

/**
 * @brief Get system uptime in milliseconds.
 * @return Uptime in milliseconds.
 */
uint32_t bsp_get_uptime_ms(void)
{
    return s_uptime_ms;
}

/**
 * @brief Get system uptime in seconds.
 * @return Uptime in seconds.
 */
uint32_t bsp_get_uptime_s(void)
{
    return s_uptime_ms / 1000U;
}

/**
 * @brief Perform controlled system reset.
 */
void bsp_system_reset(void)
{
    bsp_early_print("[BSP] System reset initiated\r\n");

    /* Disable all interrupts */
    (void)bsp_disable_interrupts();

    /* TODO: Save critical state to MRAM before reset */

    /* LEON4 reset via trap */
    /* Write to system reset register or trigger watchdog */
    volatile uint32_t *sys_reset = (volatile uint32_t *)0x80000000U;
    *sys_reset = 0U;

    /* Should not reach here */
    for (;;) {
        /* Infinite loop — processor should have reset */
    }
}

/**
 * @brief Enable global interrupts on SPARC.
 */
void bsp_enable_interrupts(void)
{
    uint32_t psr;

    /* Read PSR, set ET (Enable Traps) bit */
    __asm__ volatile (
        "rd %%psr, %0\n\t"
        "or %0, 0x20, %0\n\t"
        "wr %0, %%psr\n\t"
        "nop; nop; nop"
        : "=r" (psr)
    );
}

/**
 * @brief Disable global interrupts on SPARC.
 * @return Previous PSR value for restore.
 */
uint32_t bsp_disable_interrupts(void)
{
    uint32_t psr;
    uint32_t new_psr;

    __asm__ volatile (
        "rd %%psr, %0\n\t"
        "andn %0, 0x20, %1\n\t"
        "wr %1, %%psr\n\t"
        "nop; nop; nop"
        : "=r" (psr), "=r" (new_psr)
    );

    return psr;
}

/**
 * @brief Restore interrupt state from previous disable.
 * @param[in] state Previous PSR value.
 */
void bsp_restore_interrupts(uint32_t state)
{
    __asm__ volatile (
        "wr %0, %%psr\n\t"
        "nop; nop; nop"
        :
        : "r" (state)
    );
}

/**
 * @brief Polled-mode early print on UART0.
 * @param[in] str Null-terminated string.
 */
void bsp_early_print(const char *str)
{
    volatile uint32_t *uart_data;
    volatile uint32_t *uart_status;

    if (str == NULL) {
        return;
    }

    uart_data   = (volatile uint32_t *)(APBUART0_BASE + UART_DATA_REG);
    uart_status = (volatile uint32_t *)(APBUART0_BASE + UART_STATUS_REG);

    while (*str != '\0') {
        /* Wait until TX FIFO is not full */
        uint32_t timeout = 100000U;
        while (((*uart_status & UART_STATUS_TE) == 0U) && (timeout > 0U)) {
            timeout--;
        }
        if (timeout == 0U) {
            return; /* TX stuck, bail out */
        }

        *uart_data = (uint32_t)(uint8_t)(*str);
        str++;
    }
}

/**
 * @brief Read OBC board temperature (placeholder — reads from I2C sensor).
 * @return Temperature in Celsius, or INT32_MIN on error.
 */
int32_t bsp_read_temperature(void)
{
    /* Temperature sensor typically at I2C address 0x48 (LM75/TMP102) */
    /* Placeholder: return a nominal value until I2C driver is ready   */
    return 25; /* 25°C nominal */
}

/**
 * @brief Enable ECC scrubbing on SRAM.
 * @return BSP_OK on success.
 */
int32_t bsp_enable_ecc_scrub(void)
{
    volatile uint32_t *ft_status;

    ft_status = (volatile uint32_t *)(FTAHBRAM_BASE + FTAHBRAM_STATUS);

    /* Enable ECC scrubbing and correctable error interrupt */
    *ft_status = FTAHBRAM_ECC_SCRUB | FTAHBRAM_ECC_CE_IRQ | FTAHBRAM_ECC_UE_IRQ;

    /* Verify setting took effect */
    uint32_t val = *ft_status;
    if ((val & FTAHBRAM_ECC_SCRUB) == 0U) {
        return BSP_ERR_MEM;
    }

    return BSP_OK;
}

/* ======================================================================
 * Private Functions
 * ====================================================================== */

/**
 * @brief Configure the GR740 PLL for 50 MHz system clock.
 * @return BSP_OK on success.
 */
static int32_t bsp_init_clock(void)
{
    /*
     * GR740 clock configuration:
     * - System clock target: 50 MHz
     * - The GR740 uses a clock gating unit (CGU) and PLL
     * - Base oscillator: typically 25 MHz or 50 MHz crystal
     *
     * For 50 MHz from 25 MHz crystal:
     *   PLL multiplier = 2, divider = 1
     *
     * Register: Clock Gating Unit at base 0x80006000
     */

    /* GR740 CGU base address */
    volatile uint32_t *cgu_unlock  = (volatile uint32_t *)0x80006000U;
    volatile uint32_t *cgu_clk_en  = (volatile uint32_t *)0x80006004U;
    volatile uint32_t *cgu_core_en = (volatile uint32_t *)0x80006008U;

    /* Unlock clock configuration */
    *cgu_unlock = 0x00000001U;

    /* Enable all core clocks (4 LEON4 cores) */
    *cgu_core_en = 0x0000000FU;

    /* Enable peripheral clocks: UART, CAN, SpW, Timer, SPI, I2C, GPIO */
    *cgu_clk_en = 0xFFFFFFFFU;

    /* Lock clock configuration */
    *cgu_unlock = 0x00000000U;

    /* Small delay for PLL to stabilize (spin-wait ~1000 cycles) */
    volatile uint32_t delay;
    for (delay = 0U; delay < 1000U; delay++) {
        __asm__ volatile ("nop");
    }

    return BSP_OK;
}

/**
 * @brief Configure SRAM memory controller.
 * @return BSP_OK on success.
 */
static int32_t bsp_init_memory(void)
{
    /*
     * Configure memory controller for 32 MB SRAM
     * - 32-bit bus width
     * - 0 wait states (SRAM is fast enough at 50 MHz)
     * - Enable ECC (4-bit EDAC)
     */

    volatile uint32_t *mcfg1 = (volatile uint32_t *)(MEMCTRL_BASE + MEMCTRL_MCFG1);
    volatile uint32_t *mcfg2 = (volatile uint32_t *)(MEMCTRL_BASE + MEMCTRL_MCFG2);

    /* MCFG1: PROM configuration — not modified, use boot defaults */
    (void)mcfg1;

    /* MCFG2: SRAM/SDRAM configuration
     * Bits [5:0]:   SRAM bank size (32 MB = encoded as 5)
     * Bits [7:6]:   Bus width (00 = 32-bit)
     * Bits [11:8]:  Read wait states (0)
     * Bits [15:12]: Write wait states (0)
     * Bit  [16]:    EDAC enable
     */
    uint32_t mcfg2_val = *mcfg2;
    mcfg2_val &= 0xFFFE0000U;          /* Clear lower bits */
    mcfg2_val |= (5U << 0);            /* Bank size: 32 MB */
    mcfg2_val |= (0U << 6);            /* 32-bit bus width */
    mcfg2_val |= (0U << 8);            /* 0 read wait states */
    mcfg2_val |= (0U << 12);           /* 0 write wait states */
    mcfg2_val |= (1U << 16);           /* Enable EDAC */
    *mcfg2 = mcfg2_val;

    /* Zero BSS section */
    extern uint32_t _bss_start;
    extern uint32_t _bss_end;
    uint32_t *bss_ptr = &_bss_start;
    while (bss_ptr < &_bss_end) {
        *bss_ptr = 0U;
        bss_ptr++;
    }

    /* Copy .data from MRAM (load address) to SRAM (runtime address) */
    extern uint32_t _data_start;
    extern uint32_t _data_end;
    extern uint32_t _data_load_start;
    uint32_t *dst = &_data_start;
    const uint32_t *src = &_data_load_start;
    while (dst < &_data_end) {
        *dst = *src;
        dst++;
        src++;
    }

    return BSP_OK;
}

/**
 * @brief Initialize UART0 for debug console at 115200 baud.
 * @return BSP_OK on success.
 */
static int32_t bsp_init_uart0(void)
{
    volatile uint32_t *uart_ctrl;
    volatile uint32_t *uart_scaler;
    volatile uint32_t *uart_status;

    uart_ctrl   = (volatile uint32_t *)(APBUART0_BASE + UART_CTRL_REG);
    uart_scaler = (volatile uint32_t *)(APBUART0_BASE + UART_SCALER_REG);
    uart_status = (volatile uint32_t *)(APBUART0_BASE + UART_STATUS_REG);

    /* Disable UART during configuration */
    *uart_ctrl = 0U;

    /*
     * Baud rate scaler calculation:
     *   scaler = (sys_clk / (baudrate * 8 + 7)) - 1
     *   For 50 MHz / 115200: scaler = (50000000 / (115200*8+7)) - 1 ≈ 53
     */
    uint32_t scaler_val = (SYS_CLK_HZ / (UART_BAUDRATE * 8U + 7U)) - 1U;
    *uart_scaler = scaler_val;

    /* Clear any pending status flags */
    (void)*uart_status;

    /* Enable transmitter and receiver, no interrupts for early console */
    *uart_ctrl = UART_CTRL_TE | UART_CTRL_RE;

    return BSP_OK;
}

/**
 * @brief Initialize the IRQAMP multi-processor interrupt controller.
 * @return BSP_OK on success.
 */
static int32_t bsp_init_irqamp(void)
{
    /* Clear all pending interrupts */
    REG_WRITE(IRQAMP_BASE, IRQAMP_ICLEAR, 0xFFFFFFFFU);

    /* Clear all force registers */
    REG_WRITE(IRQAMP_BASE, IRQAMP_IFORCE, 0x00000000U);

    /* Set all interrupts to level 1 (default priority) */
    REG_WRITE(IRQAMP_BASE, IRQAMP_ILEVEL, 0x00000000U);

    /* Mask all interrupts on CPU0 initially — will be unmasked per driver */
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU0, 0x00000000U);

    /* Disable interrupts on CPUs 1-3 (single core usage for now) */
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU1, 0x00000000U);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU2, 0x00000000U);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU3, 0x00000000U);

    return BSP_OK;
}

/**
 * @brief Initialize the system tick timer (1 ms period).
 * @return BSP_OK on success.
 */
static int32_t bsp_init_systick(void)
{
    /*
     * GPTIMER setup for 1 ms tick:
     * - Scaler: divide system clock to 1 MHz base (prescaler = 49)
     * - Timer 0: count 1000 ticks (= 1 ms at 1 MHz)
     * - Auto-restart, interrupt enabled
     */

    /* Set scaler reload to achieve 1 MHz timer clock */
    uint32_t scaler_reload = (SYS_CLK_HZ / 1000000U) - 1U; /* = 49 */
    REG_WRITE(GPTIMER0_BASE, GPTIMER_SRELOAD, scaler_reload);
    REG_WRITE(GPTIMER0_BASE, GPTIMER_SCALER, scaler_reload);

    /* Configure Timer 0 for 1 ms period */
    uint32_t timer_reload = 1000U - 1U; /* 1000 ticks at 1 MHz = 1 ms */
    REG_WRITE(GPTIMER0_BASE, GPTIMER_T0_RELOAD, timer_reload);
    REG_WRITE(GPTIMER0_BASE, GPTIMER_T0_COUNTER, timer_reload);

    /* Register systick ISR */
    int32_t rc = irq_register(IRQ_TIMER0, bsp_systick_isr);
    if (rc != 0) {
        return BSP_ERR_CLK;
    }

    /* Enable Timer 0 interrupt in IRQAMP */
    irq_enable(IRQ_TIMER0);

    /* Start Timer 0: enable + restart + interrupt enable + load */
    REG_WRITE(GPTIMER0_BASE, GPTIMER_T0_CTRL,
              GPTIMER_CTRL_EN | GPTIMER_CTRL_RS |
              GPTIMER_CTRL_IE | GPTIMER_CTRL_LD);

    return BSP_OK;
}

/**
 * @brief Initialize MRAM storage via SPI.
 * @return BSP_OK on success.
 */
static int32_t bsp_init_mram(void)
{
    /*
     * MRAM is accessible via the SPI controller (SPICTRL).
     * At boot, we configure the SPI for MRAM access and verify
     * that the MRAM is responsive by reading the status register.
     *
     * Full SPI init is handled by the SPI driver; here we just
     * ensure the MRAM chip select is configured and do a basic
     * connectivity check.
     */

    volatile uint32_t *spi_mode = (volatile uint32_t *)(SPICTRL0_BASE + SPICTRL_MODE);
    volatile uint32_t *spi_tx   = (volatile uint32_t *)(SPICTRL0_BASE + SPICTRL_TX);
    volatile uint32_t *spi_rx   = (volatile uint32_t *)(SPICTRL0_BASE + SPICTRL_RX);
    volatile uint32_t *spi_evt  = (volatile uint32_t *)(SPICTRL0_BASE + SPICTRL_EVENT);
    volatile uint32_t *spi_sel  = (volatile uint32_t *)(SPICTRL0_BASE + SPICTRL_SLVSEL);

    /* Configure SPI: Master mode, CPOL=0, CPHA=0, 8-bit words */
    /* Prescaler for ~10 MHz SPI clock from 50 MHz system:
     * f_spi = f_sys / (4 * (PM + 1))
     * 10 MHz = 50 MHz / (4 * (PM+1)) → PM = 0 (gives 12.5 MHz, close enough)
     */
    uint32_t mode_val = SPICTRL_MODE_MS | SPICTRL_MODE_EN;
    mode_val |= (7U << SPICTRL_MODE_LEN_SHIFT);  /* 8-bit word length (LEN=7) */
    mode_val |= (0U << SPICTRL_MODE_PM_SHIFT);    /* PM=0 for max speed */
    *spi_mode = mode_val;

    /* Select MRAM chip (slave 0) */
    *spi_sel = 0x00000001U;

    /* Send MRAM Read Status Register command (0x05) */
    *spi_tx = 0x05U;

    /* Wait for transfer complete */
    uint32_t timeout = 10000U;
    while (((*spi_evt & SPICTRL_EVENT_LT) == 0U) && (timeout > 0U)) {
        timeout--;
    }
    if (timeout == 0U) {
        *spi_sel = 0x00000000U; /* Deselect */
        return BSP_ERR_NVM;
    }

    /* Read status byte */
    uint32_t status = *spi_rx;
    (void)status; /* Status check can be expanded */

    /* Clear event flags */
    *spi_evt = 0xFFFFFFFFU;

    /* Deselect MRAM */
    *spi_sel = 0x00000000U;

    return BSP_OK;
}

/**
 * @brief System tick ISR — increments uptime counter.
 *
 * Called every 1 ms by GPTIMER Timer 0 interrupt.
 * Minimal processing to keep ISR latency low.
 */
static void bsp_systick_isr(void)
{
    /* Increment uptime counter */
    s_uptime_ms++;

    /* Clear timer interrupt pending flag */
    REG_WRITE(GPTIMER0_BASE, GPTIMER_T0_CTRL,
              REG_READ(GPTIMER0_BASE, GPTIMER_T0_CTRL) | GPTIMER_CTRL_IP);
}
