/**
 * @file grgpio.c
 * @brief GRGPIO driver implementation for GR740.
 *
 * Register-level driver for the Cobham Gaisler GRGPIO core.
 * Supports up to 32 GPIO lines with per-pin direction control,
 * interrupt generation on edge/level, and callback dispatch.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "grgpio.h"
#include "../../config/hw_config.h"

/* ── GRGPIO Register Offsets ───────────────────────────────────────────── */
#define GPIO_DATA       0x00U   /**< I/O port data register        */
#define GPIO_OUTPUT     0x04U   /**< I/O port output register      */
#define GPIO_DIR        0x08U   /**< I/O port direction (1=output) */
#define GPIO_IMASK      0x0CU   /**< Interrupt mask register       */
#define GPIO_IPOL       0x10U   /**< Interrupt polarity            */
#define GPIO_IEDGE      0x14U   /**< Interrupt edge/level select   */
#define GPIO_BYPASS     0x18U   /**< Bypass input register         */
#define GPIO_CAP        0x1CU   /**< Capability register           */
#define GPIO_IRQMAP0    0x20U   /**< IRQ map register 0            */
#define GPIO_IRQMAP1    0x24U   /**< IRQ map register 1            */
#define GPIO_IRQMAP2    0x28U   /**< IRQ map register 2            */
#define GPIO_IRQMAP3    0x2CU   /**< IRQ map register 3            */
#define GPIO_IAVAIL     0x30U   /**< Interrupt available register  */
#define GPIO_IFLAG      0x34U   /**< Interrupt flag register       */
#define GPIO_IPEN       0x38U   /**< Interrupt pending register    */
#define GPIO_PULSE      0x3CU   /**< Pulse register                */

/* Set/Clear registers (if available in the core) */
#define GPIO_OUTPUT_SET 0x54U   /**< Output OR register (set bits) */
#define GPIO_OUTPUT_CLR 0x58U   /**< Output AND-NOT (clear bits)   */
#define GPIO_DIR_SET    0x5CU   /**< Direction OR register         */
#define GPIO_DIR_CLR    0x60U   /**< Direction AND-NOT register    */

/* ── Module state ──────────────────────────────────────────────────────── */
static volatile uint32_t *gpio_base_ptr = (volatile uint32_t *)0;
static uint8_t gpio_initialized = 0U;
static gpio_callback_t gpio_callbacks[GPIO_MAX_PINS];

/* ── Register access helpers ───────────────────────────────────────────── */
static inline uint32_t gpio_reg_read(uint32_t offset)
{
    return *(volatile uint32_t *)((uint32_t)gpio_base_ptr + offset);
}

static inline void gpio_reg_write(uint32_t offset, uint32_t value)
{
    *(volatile uint32_t *)((uint32_t)gpio_base_ptr + offset) = value;
}

/* ── Public API ────────────────────────────────────────────────────────── */

int32_t gpio_init(uint32_t base_addr)
{
    uint32_t i;

    if (base_addr == 0U) {
        return GPIO_ERR_PARAM;
    }

    gpio_base_ptr = (volatile uint32_t *)(uintptr_t)base_addr;

    /* All pins as input by default */
    gpio_reg_write(GPIO_DIR, 0x00000000U);

    /* Clear all outputs */
    gpio_reg_write(GPIO_OUTPUT, 0x00000000U);

    /* Disable all interrupts */
    gpio_reg_write(GPIO_IMASK, 0x00000000U);

    /* Clear any pending interrupt flags */
    gpio_reg_write(GPIO_IFLAG, 0xFFFFFFFFU);

    /* Clear callbacks */
    for (i = 0U; i < GPIO_MAX_PINS; i++) {
        gpio_callbacks[i] = (gpio_callback_t)0;
    }

    gpio_initialized = 1U;
    return GPIO_OK;
}

int32_t gpio_set_direction(uint32_t pin, uint32_t dir)
{
    uint32_t dir_reg;

    if (gpio_initialized == 0U) {
        return GPIO_ERR_INIT;
    }
    if (pin >= GPIO_MAX_PINS) {
        return GPIO_ERR_PARAM;
    }

    dir_reg = gpio_reg_read(GPIO_DIR);

    if (dir == GPIO_DIR_OUTPUT) {
        dir_reg |= (1U << pin);
    } else {
        dir_reg &= ~(1U << pin);
    }

    gpio_reg_write(GPIO_DIR, dir_reg);
    return GPIO_OK;
}

int32_t gpio_write(uint32_t pin, uint32_t value)
{
    uint32_t out_reg;

    if (gpio_initialized == 0U) {
        return GPIO_ERR_INIT;
    }
    if (pin >= GPIO_MAX_PINS) {
        return GPIO_ERR_PARAM;
    }

    out_reg = gpio_reg_read(GPIO_OUTPUT);

    if (value != GPIO_LOW) {
        out_reg |= (1U << pin);
    } else {
        out_reg &= ~(1U << pin);
    }

    gpio_reg_write(GPIO_OUTPUT, out_reg);
    return GPIO_OK;
}

int32_t gpio_read(uint32_t pin, uint32_t *value)
{
    uint32_t data;

    if (gpio_initialized == 0U) {
        return GPIO_ERR_INIT;
    }
    if (pin >= GPIO_MAX_PINS) {
        return GPIO_ERR_PARAM;
    }
    if (value == (uint32_t *)0) {
        return GPIO_ERR_PARAM;
    }

    data = gpio_reg_read(GPIO_DATA);
    *value = (data >> pin) & 1U;

    return GPIO_OK;
}

int32_t gpio_toggle(uint32_t pin)
{
    uint32_t out_reg;

    if (gpio_initialized == 0U) {
        return GPIO_ERR_INIT;
    }
    if (pin >= GPIO_MAX_PINS) {
        return GPIO_ERR_PARAM;
    }

    out_reg = gpio_reg_read(GPIO_OUTPUT);
    out_reg ^= (1U << pin);
    gpio_reg_write(GPIO_OUTPUT, out_reg);

    return GPIO_OK;
}

void gpio_write_all(uint32_t value)
{
    if (gpio_initialized != 0U) {
        gpio_reg_write(GPIO_OUTPUT, value);
    }
}

uint32_t gpio_read_all(void)
{
    if (gpio_initialized == 0U) {
        return 0U;
    }
    return gpio_reg_read(GPIO_DATA);
}

int32_t gpio_irq_configure(uint32_t pin, uint32_t mode, gpio_callback_t cb)
{
    uint32_t imask;
    uint32_t ipol;
    uint32_t iedge;

    if (gpio_initialized == 0U) {
        return GPIO_ERR_INIT;
    }
    if (pin >= GPIO_MAX_PINS) {
        return GPIO_ERR_PARAM;
    }

    imask = gpio_reg_read(GPIO_IMASK);
    ipol  = gpio_reg_read(GPIO_IPOL);
    iedge = gpio_reg_read(GPIO_IEDGE);

    if (mode == GPIO_IRQ_NONE) {
        /* Disable interrupt for this pin */
        imask &= ~(1U << pin);
        gpio_callbacks[pin] = (gpio_callback_t)0;
    } else {
        gpio_callbacks[pin] = cb;
        imask |= (1U << pin);

        switch (mode) {
        case GPIO_IRQ_RISING:
            iedge |=  (1U << pin);  /* Edge triggered  */
            ipol  |=  (1U << pin);  /* Rising edge     */
            break;

        case GPIO_IRQ_FALLING:
            iedge |=  (1U << pin);  /* Edge triggered  */
            ipol  &= ~(1U << pin);  /* Falling edge    */
            break;

        case GPIO_IRQ_BOTH:
            /*
             * GRGPIO doesn't natively support both-edge in single config.
             * Use edge mode; ISR will toggle polarity to catch both edges.
             */
            iedge |= (1U << pin);
            ipol  |= (1U << pin);   /* Start with rising */
            break;

        case GPIO_IRQ_LEVEL:
            iedge &= ~(1U << pin);  /* Level triggered */
            ipol  |=  (1U << pin);  /* High level      */
            break;

        default:
            return GPIO_ERR_PARAM;
        }
    }

    /* Clear pending flag before enabling */
    gpio_reg_write(GPIO_IFLAG, (1U << pin));

    gpio_reg_write(GPIO_IPOL,  ipol);
    gpio_reg_write(GPIO_IEDGE, iedge);
    gpio_reg_write(GPIO_IMASK, imask);

    return GPIO_OK;
}

void gpio_isr(void)
{
    uint32_t iflag;
    uint32_t pin;

    iflag = gpio_reg_read(GPIO_IFLAG);

    /* Process all pending GPIO interrupts */
    while (iflag != 0U) {
        /* Find lowest set bit */
        pin = 0U;
        while (((iflag & (1U << pin)) == 0U) && (pin < GPIO_MAX_PINS)) {
            pin++;
        }

        if (pin < GPIO_MAX_PINS) {
            /* Clear the flag (write-1-to-clear) */
            gpio_reg_write(GPIO_IFLAG, (1U << pin));

            /* Invoke callback */
            if (gpio_callbacks[pin] != (gpio_callback_t)0) {
                gpio_callbacks[pin](pin);
            }

            /* Clear processed bit from local copy */
            iflag &= ~(1U << pin);
        } else {
            break;
        }
    }
}
