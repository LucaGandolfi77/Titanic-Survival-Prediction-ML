/**
 * @file irq_handler.h
 * @brief IRQAMP interrupt controller driver interface for GR740.
 *
 * Provides interrupt registration, enable/disable, and dispatch
 * for the GR740 AMBA IRQAMP multi-processor interrupt controller.
 *
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef IRQ_HANDLER_H
#define IRQ_HANDLER_H

#include <stdint.h>

/* ======================================================================
 * Constants
 * ====================================================================== */
#define IRQ_MAX_SOURCES         32U     /**< Maximum IRQ sources         */
#define IRQ_PRIORITY_MIN        1U      /**< Minimum priority (lowest)   */
#define IRQ_PRIORITY_MAX        15U     /**< Maximum priority (highest)  */

/* ======================================================================
 * Types
 * ====================================================================== */

/** Interrupt service routine function pointer type */
typedef void (*irq_handler_fn)(void);

/* ======================================================================
 * Function Prototypes
 * ====================================================================== */

/**
 * @brief Initialize the IRQAMP interrupt controller.
 *
 * Clears all pending interrupts, masks all sources, and initializes
 * the ISR dispatch table.
 *
 * @return 0 on success, negative on error.
 */
int32_t irq_init(void);

/**
 * @brief Register an ISR for a specific IRQ source.
 * @param[in] irq_num  IRQ number (0–31).
 * @param[in] handler  Function pointer to ISR.
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_register(uint32_t irq_num, irq_handler_fn handler);

/**
 * @brief Unregister an ISR for a specific IRQ source.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_unregister(uint32_t irq_num);

/**
 * @brief Enable (unmask) a specific interrupt source.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_enable(uint32_t irq_num);

/**
 * @brief Disable (mask) a specific interrupt source.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_disable(uint32_t irq_num);

/**
 * @brief Set the priority level for an interrupt source.
 * @param[in] irq_num  IRQ number (0–31).
 * @param[in] priority Priority level (1–15, 15=highest).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_set_priority(uint32_t irq_num, uint8_t priority);

/**
 * @brief Top-level interrupt dispatcher.
 *
 * Called from the SPARC trap handler (trap 0x11–0x1F).
 * Reads the pending interrupt register, identifies the source,
 * and dispatches to the registered handler.
 *
 * @param[in] irq_num The hardware IRQ number from the trap.
 */
void irq_dispatch(uint32_t irq_num);

/**
 * @brief Force a specific interrupt (for testing).
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_force(uint32_t irq_num);

/**
 * @brief Clear a pending interrupt.
 * @param[in] irq_num IRQ number (0–31).
 * @return 0 on success, -1 on invalid parameters.
 */
int32_t irq_clear(uint32_t irq_num);

#endif /* IRQ_HANDLER_H */
