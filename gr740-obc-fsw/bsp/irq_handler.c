/**
 * @file irq_handler.c
 * @brief IRQAMP interrupt controller driver for GR740 LEON4FT.
 *
 * Implements interrupt registration, enable/disable, priority management,
 * and dispatch for the AMBA IRQAMP controller with support for up to 32
 * interrupt sources and 4 processor cores.
 *
 * @reference GR740 User Manual, GRLIB IRQAMP IP Core documentation
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#include "irq_handler.h"
#include "../config/hw_config.h"

#include <stdint.h>
#include <stddef.h>

/* ======================================================================
 * Private Data
 * ====================================================================== */

/** ISR dispatch table — one handler per IRQ source */
static irq_handler_fn s_isr_table[IRQ_MAX_SOURCES];

/** IRQ invocation counter (for diagnostics) */
static volatile uint32_t s_irq_count[IRQ_MAX_SOURCES];

/** Flag indicating IRQ subsystem is initialized */
static volatile uint8_t s_irq_initialized = 0U;

/* ======================================================================
 * Public Functions
 * ====================================================================== */

/**
 * @brief Initialize the IRQAMP interrupt controller.
 * @return 0 on success, negative on error.
 */
int32_t irq_init(void)
{
    uint32_t i;

    /* Initialize ISR dispatch table */
    for (i = 0U; i < IRQ_MAX_SOURCES; i++) {
        s_isr_table[i] = NULL;
        s_irq_count[i] = 0U;
    }

    /* Clear all pending interrupts */
    REG_WRITE(IRQAMP_BASE, IRQAMP_ICLEAR, 0xFFFFFFFFU);

    /* Clear all force registers */
    REG_WRITE(IRQAMP_BASE, IRQAMP_IFORCE, 0x00000000U);

    /* Mask all interrupts on all CPUs */
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU0, 0x00000000U);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU1, 0x00000000U);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU2, 0x00000000U);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU3, 0x00000000U);

    /* Set all interrupts to level 0 (default) */
    REG_WRITE(IRQAMP_BASE, IRQAMP_ILEVEL, 0x00000000U);

    s_irq_initialized = 1U;

    return 0;
}

/**
 * @brief Register an ISR for a specific IRQ source.
 * @param[in] irq_num  IRQ number (0–31).
 * @param[in] handler  Function pointer to ISR (must not be NULL).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_register(uint32_t irq_num, irq_handler_fn handler)
{
    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }
    if (handler == NULL) {
        return -1;
    }

    s_isr_table[irq_num] = handler;
    s_irq_count[irq_num] = 0U;

    return 0;
}

/**
 * @brief Unregister an ISR for a specific IRQ source.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_unregister(uint32_t irq_num)
{
    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }

    /* Disable the interrupt before removing handler */
    (void)irq_disable(irq_num);

    s_isr_table[irq_num] = NULL;

    return 0;
}

/**
 * @brief Enable (unmask) a specific interrupt source on CPU0.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_enable(uint32_t irq_num)
{
    uint32_t mask;

    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }

    mask = REG_READ(IRQAMP_BASE, IRQAMP_IMASK_CPU0);
    mask |= (1U << irq_num);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU0, mask);

    return 0;
}

/**
 * @brief Disable (mask) a specific interrupt source on CPU0.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_disable(uint32_t irq_num)
{
    uint32_t mask;

    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }

    mask = REG_READ(IRQAMP_BASE, IRQAMP_IMASK_CPU0);
    mask &= ~(1U << irq_num);
    REG_WRITE(IRQAMP_BASE, IRQAMP_IMASK_CPU0, mask);

    return 0;
}

/**
 * @brief Set the priority level for an interrupt source.
 * @param[in] irq_num  IRQ number (0–31).
 * @param[in] priority Priority level (1–15).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_set_priority(uint32_t irq_num, uint8_t priority)
{
    uint32_t ilevel;

    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }
    if ((priority < IRQ_PRIORITY_MIN) || (priority > IRQ_PRIORITY_MAX)) {
        return -1;
    }

    /*
     * IRQAMP level register: bit N=1 means IRQ N is level 1 (high priority)
     * For finer control, extended IRQAMP registers would be used.
     * Here we use the basic level bit: 1=high, 0=low.
     */
    ilevel = REG_READ(IRQAMP_BASE, IRQAMP_ILEVEL);
    if (priority >= 8U) {
        ilevel |= (1U << irq_num);     /* High priority */
    } else {
        ilevel &= ~(1U << irq_num);    /* Low priority */
    }
    REG_WRITE(IRQAMP_BASE, IRQAMP_ILEVEL, ilevel);

    return 0;
}

/**
 * @brief Top-level interrupt dispatcher.
 * @param[in] irq_num The hardware IRQ number from the trap.
 */
void irq_dispatch(uint32_t irq_num)
{
    if (irq_num >= IRQ_MAX_SOURCES) {
        return;
    }

    /* Increment counter */
    s_irq_count[irq_num]++;

    /* Dispatch to registered handler */
    if (s_isr_table[irq_num] != NULL) {
        s_isr_table[irq_num]();
    }

    /* Clear the interrupt */
    REG_WRITE(IRQAMP_BASE, IRQAMP_ICLEAR, (1U << irq_num));
}

/**
 * @brief Force a specific interrupt.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_force(uint32_t irq_num)
{
    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }

    REG_WRITE(IRQAMP_BASE, IRQAMP_IFORCE_CPU0, (1U << irq_num));

    return 0;
}

/**
 * @brief Clear a pending interrupt.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_clear(uint32_t irq_num)
{
    if (irq_num >= IRQ_MAX_SOURCES) {
        return -1;
    }

    REG_WRITE(IRQAMP_BASE, IRQAMP_ICLEAR, (1U << irq_num));

    return 0;
}
