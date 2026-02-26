/**
 * @file hw_config.h
 * @brief Hardware register base addresses and constants for GR740 SoC.
 *
 * All addresses are based on the GR740 User Manual and GRLIB IP Core
 * User Manual. Registers are accessed via volatile uint32_t pointers.
 *
 * @reference GR740 User Manual (GR740-UM), GRLIB IP Core User Manual
 * @author OBC Flight Software Team
 * @version 1.0.0
 * @date 2026-02-26
 *
 * Copyright (C) 2026 — ESA Public License v2.0
 */

#ifndef HW_CONFIG_H
#define HW_CONFIG_H

#include <stdint.h>

/* ======================================================================
 * SYSTEM CLOCK
 * ====================================================================== */
#define SYS_CLK_HZ             50000000U    /**< System clock: 50 MHz */
#define SYS_CLK_MHZ            50U

/* ======================================================================
 * AMBA AHB/APB BASE ADDRESSES
 * Derived from GR740 AMBA Plug-and-Play area
 * ====================================================================== */

/** APB bridge base */
#define APB_BASE                0x80000000U

/* ======================================================================
 * APBUART (UART) — up to 6 instances on GR740
 * Each UART has 5 registers at 0x04 spacing
 * ====================================================================== */
#define APBUART0_BASE           0x80000100U
#define APBUART1_BASE           0x80000200U
#define APBUART2_BASE           0x80000300U
#define APBUART3_BASE           0x80000400U
#define APBUART4_BASE           0x80000500U
#define APBUART5_BASE           0x80000600U

/* APBUART register offsets */
#define UART_DATA_REG           0x00U   /**< Data register               */
#define UART_STATUS_REG         0x04U   /**< Status register             */
#define UART_CTRL_REG           0x08U   /**< Control register            */
#define UART_SCALER_REG         0x0CU   /**< Scaler/baud rate register   */
#define UART_FIFO_REG           0x10U   /**< FIFO debug register         */

/* UART status register bits */
#define UART_STATUS_DR          (1U << 0)   /**< Data ready              */
#define UART_STATUS_TS          (1U << 1)   /**< Transmitter shift empty */
#define UART_STATUS_TE          (1U << 2)   /**< Transmitter FIFO empty  */
#define UART_STATUS_BR          (1U << 3)   /**< Break received          */
#define UART_STATUS_OV          (1U << 4)   /**< Overrun                 */
#define UART_STATUS_PE          (1U << 5)   /**< Parity error            */
#define UART_STATUS_FE          (1U << 6)   /**< Framing error           */
#define UART_STATUS_TH          (1U << 7)   /**< TX FIFO half empty      */
#define UART_STATUS_RH          (1U << 8)   /**< RX FIFO half full       */
#define UART_STATUS_TF          (1U << 9)   /**< TX FIFO full            */
#define UART_STATUS_RF          (1U << 10)  /**< RX FIFO full            */

/* UART control register bits */
#define UART_CTRL_RE            (1U << 0)   /**< Receiver enable         */
#define UART_CTRL_TE            (1U << 1)   /**< Transmitter enable      */
#define UART_CTRL_RI            (1U << 2)   /**< Receiver interrupt en   */
#define UART_CTRL_TI            (1U << 3)   /**< Transmitter int enable  */
#define UART_CTRL_PS            (1U << 4)   /**< Parity select (odd)     */
#define UART_CTRL_PE            (1U << 5)   /**< Parity enable           */
#define UART_CTRL_FL            (1U << 6)   /**< Flow control enable     */
#define UART_CTRL_LB            (1U << 7)   /**< Loopback mode           */
#define UART_CTRL_DB            (1U << 11)  /**< Debug mode (FIFO)       */
#define UART_CTRL_FA            (1U << 31)  /**< FIFO available          */

/* ======================================================================
 * IRQAMP — Multi-processor interrupt controller
 * ====================================================================== */
#define IRQAMP_BASE             0x80000200U

/* IRQAMP register offsets */
#define IRQAMP_ILEVEL           0x00U   /**< Interrupt level register    */
#define IRQAMP_IPEND            0x04U   /**< Interrupt pending register  */
#define IRQAMP_IFORCE           0x08U   /**< Interrupt force register    */
#define IRQAMP_ICLEAR           0x0CU   /**< Interrupt clear register    */
#define IRQAMP_MPSTAT           0x10U   /**< MP status register          */
#define IRQAMP_BCAST            0x14U   /**< Broadcast register          */
#define IRQAMP_IMASK_CPU0       0x40U   /**< CPU0 interrupt mask         */
#define IRQAMP_IMASK_CPU1       0x44U   /**< CPU1 interrupt mask         */
#define IRQAMP_IMASK_CPU2       0x48U   /**< CPU2 interrupt mask         */
#define IRQAMP_IMASK_CPU3       0x4CU   /**< CPU3 interrupt mask         */
#define IRQAMP_IFORCE_CPU0      0x80U   /**< CPU0 interrupt force        */

/* ======================================================================
 * GPTIMER — General Purpose Timer Unit
 * GR740 has multiple timer units; primary at GPTIMER0_BASE
 * Each unit has: scaler, scaler_reload, config, N×(counter,reload,ctrl)
 * ====================================================================== */
#define GPTIMER0_BASE           0x80000300U
#define GPTIMER1_BASE           0x80000400U

/* GPTIMER register offsets (from base) */
#define GPTIMER_SCALER          0x00U   /**< Scaler value register       */
#define GPTIMER_SRELOAD         0x04U   /**< Scaler reload register      */
#define GPTIMER_CONFIG          0x08U   /**< Configuration register      */
#define GPTIMER_LATCH_CFG       0x0CU   /**< Latch configuration         */

/* Timer N registers (offset = 0x10 + N*0x10) */
#define GPTIMER_T0_COUNTER      0x10U   /**< Timer 0 counter value       */
#define GPTIMER_T0_RELOAD       0x14U   /**< Timer 0 reload value        */
#define GPTIMER_T0_CTRL         0x18U   /**< Timer 0 control register    */
#define GPTIMER_T0_LATCH        0x1CU   /**< Timer 0 latch register      */

/* Timer control register bits */
#define GPTIMER_CTRL_EN         (1U << 0)   /**< Enable timer            */
#define GPTIMER_CTRL_RS         (1U << 1)   /**< Restart on underflow    */
#define GPTIMER_CTRL_LD         (1U << 2)   /**< Load counter            */
#define GPTIMER_CTRL_IE         (1U << 3)   /**< Interrupt enable        */
#define GPTIMER_CTRL_IP         (1U << 4)   /**< Interrupt pending       */
#define GPTIMER_CTRL_CH         (1U << 5)   /**< Chain with preceding    */
#define GPTIMER_CTRL_DH         (1U << 6)   /**< Debug halt              */

/* ======================================================================
 * GRCAN — CAN 2.0B Controller
 * ====================================================================== */
#define GRCAN0_BASE             0x80000700U
#define GRCAN1_BASE             0x80000800U

/* GRCAN register offsets */
#define GRCAN_CONF              0x00U   /**< Configuration register      */
#define GRCAN_STAT              0x04U   /**< Status register             */
#define GRCAN_CTRL              0x08U   /**< Control register            */
#define GRCAN_SMASK             0x0CU   /**< Status mask register        */
#define GRCAN_PIMSR             0x10U   /**< Primary IRQ mask/status     */
#define GRCAN_PIMR              0x14U   /**< Primary IRQ mask register   */
#define GRCAN_PISR              0x18U   /**< Primary IRQ status register */
#define GRCAN_IMR               0x1CU   /**< IRQ mask register           */
#define GRCAN_PICR              0x20U   /**< Primary IRQ clear register  */
#define GRCAN_TX_CTRL           0x200U  /**< TX control register         */
#define GRCAN_TX_ADDR           0x204U  /**< TX descriptor base address  */
#define GRCAN_TX_SIZE           0x208U  /**< TX descriptor ring size     */
#define GRCAN_TX_WR             0x20CU  /**< TX write pointer            */
#define GRCAN_TX_RD             0x210U  /**< TX read pointer             */
#define GRCAN_TX_IRQ            0x214U  /**< TX IRQ threshold            */
#define GRCAN_RX_CTRL           0x300U  /**< RX control register         */
#define GRCAN_RX_ADDR           0x304U  /**< RX descriptor base address  */
#define GRCAN_RX_SIZE           0x308U  /**< RX descriptor ring size     */
#define GRCAN_RX_WR             0x30CU  /**< RX write pointer            */
#define GRCAN_RX_RD             0x310U  /**< RX read pointer             */
#define GRCAN_RX_IRQ            0x314U  /**< RX IRQ threshold            */

/* GRCAN control register bits */
#define GRCAN_CTRL_RESET        (1U << 0)   /**< Reset CAN core          */
#define GRCAN_CTRL_ENABLE       (1U << 1)   /**< Enable CAN core         */
#define GRCAN_CTRL_SELECTION    (1U << 2)   /**< Selection mode           */
#define GRCAN_CTRL_SILENT       (1U << 3)   /**< Silent mode              */

/* GRCAN status register bits */
#define GRCAN_STAT_PASS         (1U << 0)   /**< Error passive            */
#define GRCAN_STAT_OFF          (1U << 1)   /**< Bus off                  */
#define GRCAN_STAT_ACTIVE       (1U << 2)   /**< Bus active               */
#define GRCAN_STAT_AHBERR       (1U << 3)   /**< AHB error                */
#define GRCAN_STAT_OR           (1U << 4)   /**< Overrun                  */
#define GRCAN_STAT_TXLOSS       (1U << 5)   /**< TX message loss          */
#define GRCAN_STAT_RXAHBERR     (1U << 6)   /**< RX AHB error             */
#define GRCAN_STAT_TXAHBERR     (1U << 7)   /**< TX AHB error             */

/* GRCAN TX/RX control bits */
#define GRCAN_TXRX_ENABLE       (1U << 0)   /**< Enable TX or RX DMA     */
#define GRCAN_TXRX_ONGOING      (1U << 1)   /**< DMA ongoing             */

/* ======================================================================
 * GRSPW2 — SpaceWire Link (4 instances on GR740)
 * ====================================================================== */
#define GRSPW0_BASE             0x80000900U
#define GRSPW1_BASE             0x80000A00U
#define GRSPW2_BASE             0x80000B00U
#define GRSPW3_BASE             0x80000C00U

/* GRSPW2 register offsets */
#define GRSPW_CTRL              0x00U   /**< Control register            */
#define GRSPW_STATUS            0x04U   /**< Status register             */
#define GRSPW_NODEADDR          0x08U   /**< Node address register       */
#define GRSPW_CLKDIV            0x0CU   /**< Clock divisor register      */
#define GRSPW_DESTKEY           0x10U   /**< Destination key register    */
#define GRSPW_TIME              0x14U   /**< Time register               */
#define GRSPW_TIMER             0x18U   /**< Timer register              */

/* DMA channel 0 registers (offset 0x20) */
#define GRSPW_DMA0_CTRL         0x20U   /**< DMA0 control register       */
#define GRSPW_DMA0_RXMAX        0x24U   /**< DMA0 max RX packet length   */
#define GRSPW_DMA0_TXDESC       0x28U   /**< DMA0 TX descriptor pointer  */
#define GRSPW_DMA0_RXDESC       0x2CU   /**< DMA0 RX descriptor pointer  */
#define GRSPW_DMA0_ADDR         0x30U   /**< DMA0 address register       */

/* GRSPW control register bits */
#define GRSPW_CTRL_LD           (1U << 0)   /**< Link disable            */
#define GRSPW_CTRL_LS           (1U << 1)   /**< Link start              */
#define GRSPW_CTRL_AS           (1U << 2)   /**< Autostart               */
#define GRSPW_CTRL_IE           (1U << 3)   /**< Interrupt enable        */
#define GRSPW_CTRL_TI           (1U << 4)   /**< Tick-in enable          */
#define GRSPW_CTRL_PM           (1U << 5)   /**< Promiscuous mode        */
#define GRSPW_CTRL_RE           (1U << 6)   /**< RMAP enable             */
#define GRSPW_CTRL_RD           (1U << 8)   /**< RMAP buffer disable     */
#define GRSPW_CTRL_TE           (1U << 9)   /**< Time RX enable          */
#define GRSPW_CTRL_TR           (1U << 10)  /**< Time TX enable          */
#define GRSPW_CTRL_RS           (1U << 6)   /**< Reset                   */

/* GRSPW status register bits */
#define GRSPW_STATUS_TO         (1U << 0)   /**< Tick out                */
#define GRSPW_STATUS_CE         (1U << 1)   /**< Credit error            */
#define GRSPW_STATUS_ER         (1U << 2)   /**< Escape error            */
#define GRSPW_STATUS_DE         (1U << 3)   /**< Disconnect error        */
#define GRSPW_STATUS_PE         (1U << 4)   /**< Parity error            */
#define GRSPW_STATUS_IA         (1U << 5)   /**< Invalid address         */
#define GRSPW_STATUS_EE         (1U << 8)   /**< Early EOP/EEP           */

/* Link state bits (in status register [23:21]) */
#define GRSPW_STATUS_LS_MASK    (0x7U << 21)
#define GRSPW_LS_ERROR_RESET    0U
#define GRSPW_LS_ERROR_WAIT     1U
#define GRSPW_LS_READY          2U
#define GRSPW_LS_STARTED        3U
#define GRSPW_LS_CONNECTING     4U
#define GRSPW_LS_RUN            5U

/* GRSPW DMA control bits */
#define GRSPW_DMA_CTRL_TE       (1U << 0)   /**< TX enable               */
#define GRSPW_DMA_CTRL_RE       (1U << 1)   /**< RX enable               */
#define GRSPW_DMA_CTRL_TI       (1U << 2)   /**< TX IRQ enable           */
#define GRSPW_DMA_CTRL_RI       (1U << 3)   /**< RX IRQ enable           */
#define GRSPW_DMA_CTRL_AI       (1U << 4)   /**< AHB error IRQ           */
#define GRSPW_DMA_CTRL_AT       (1U << 12)  /**< Abort TX                */
#define GRSPW_DMA_CTRL_RX_RST   (1U << 13)  /**< RX reset                */
#define GRSPW_DMA_CTRL_NS       (1U << 8)   /**< No spill                */
#define GRSPW_DMA_CTRL_EN       (1U << 9)   /**< Strip address           */

/* ======================================================================
 * SPICTRL — SPI Controller
 * ====================================================================== */
#define SPICTRL0_BASE           0x80000D00U

/* SPICTRL register offsets */
#define SPICTRL_CAP             0x00U   /**< Capability register         */
#define SPICTRL_MODE            0x20U   /**< Mode register               */
#define SPICTRL_EVENT           0x24U   /**< Event register              */
#define SPICTRL_MASK            0x28U   /**< Mask register               */
#define SPICTRL_CMD             0x2CU   /**< Command register            */
#define SPICTRL_TX              0x30U   /**< TX data register            */
#define SPICTRL_RX              0x34U   /**< RX data register            */
#define SPICTRL_SLVSEL          0x38U   /**< Slave select register       */

/* SPICTRL mode register bits */
#define SPICTRL_MODE_LOOP       (1U << 0)   /**< Loopback mode           */
#define SPICTRL_MODE_CPOL       (1U << 1)   /**< Clock polarity          */
#define SPICTRL_MODE_CPHA       (1U << 2)   /**< Clock phase             */
#define SPICTRL_MODE_DIV16      (1U << 3)   /**< Divide by 16            */
#define SPICTRL_MODE_REV        (1U << 4)   /**< Bit reversal            */
#define SPICTRL_MODE_MS         (1U << 5)   /**< Master/slave select     */
#define SPICTRL_MODE_EN         (1U << 6)   /**< Enable core             */
#define SPICTRL_MODE_LEN_SHIFT  20U         /**< Word length shift       */
#define SPICTRL_MODE_LEN_MASK   (0xFU << 20)
#define SPICTRL_MODE_PM_SHIFT   16U         /**< Prescale modulus shift  */
#define SPICTRL_MODE_PM_MASK    (0xFU << 16)
#define SPICTRL_MODE_CG_SHIFT   7U          /**< Clock gap shift         */
#define SPICTRL_MODE_FACT       (1U << 13)  /**< PM factor               */

/* SPICTRL event register bits */
#define SPICTRL_EVENT_LT        (1U << 14)  /**< Last character TX'd     */
#define SPICTRL_EVENT_OV        (1U << 12)  /**< Overrun                 */
#define SPICTRL_EVENT_UN        (1U << 13)  /**< Underrun                */
#define SPICTRL_EVENT_NE        (1U << 1)   /**< Not empty               */
#define SPICTRL_EVENT_NF        (1U << 2)   /**< Not full                */

/* ======================================================================
 * I2CMST — I2C Master Controller
 * ====================================================================== */
#define I2CMST0_BASE            0x80000E00U

/* I2CMST register offsets */
#define I2C_PRER_LO             0x00U   /**< Prescale register (low)     */
#define I2C_PRER_HI             0x04U   /**< Prescale register (high)    */
#define I2C_CTR                 0x08U   /**< Control register            */
#define I2C_TXR                 0x0CU   /**< Transmit register           */
#define I2C_RXR                 0x0CU   /**< Receive register (same)     */
#define I2C_CR                  0x10U   /**< Command register            */
#define I2C_SR                  0x10U   /**< Status register (same)      */

/* I2C control register bits */
#define I2C_CTR_EN              (1U << 7)   /**< Core enable             */
#define I2C_CTR_IEN             (1U << 6)   /**< Interrupt enable        */

/* I2C command register bits */
#define I2C_CR_STA              (1U << 7)   /**< Generate START          */
#define I2C_CR_STO              (1U << 6)   /**< Generate STOP           */
#define I2C_CR_RD               (1U << 5)   /**< Read from slave         */
#define I2C_CR_WR               (1U << 4)   /**< Write to slave          */
#define I2C_CR_ACK              (1U << 3)   /**< Send ACK (0) or NACK(1) */
#define I2C_CR_IACK             (1U << 0)   /**< Interrupt acknowledge   */

/* I2C status register bits */
#define I2C_SR_RXACK            (1U << 7)   /**< Received ACK            */
#define I2C_SR_BUSY             (1U << 6)   /**< Bus busy                */
#define I2C_SR_AL               (1U << 5)   /**< Arbitration lost        */
#define I2C_SR_TIP              (1U << 1)   /**< Transfer in progress    */
#define I2C_SR_IF               (1U << 0)   /**< Interrupt flag          */

/* ======================================================================
 * GRGPIO — General Purpose I/O
 * ====================================================================== */
#define GRGPIO0_BASE            0x80000F00U

/* GRGPIO register offsets */
#define GPIO_DATA               0x00U   /**< I/O port data register      */
#define GPIO_OUTPUT             0x04U   /**< I/O port output register    */
#define GPIO_DIR                0x08U   /**< I/O port direction register */
#define GPIO_IMASK              0x0CU   /**< Interrupt mask register     */
#define GPIO_IPOL               0x10U   /**< Interrupt polarity register */
#define GPIO_IEDGE              0x14U   /**< Interrupt edge register     */
#define GPIO_CAP                0x18U   /**< Port capability register    */
#define GPIO_IRQMAP0            0x20U   /**< IRQ map register 0          */
#define GPIO_IRQMAP1            0x24U   /**< IRQ map register 1          */
#define GPIO_IRQMAP2            0x28U   /**< IRQ map register 2          */
#define GPIO_IRQMAP3            0x2CU   /**< IRQ map register 3          */

/* ======================================================================
 * MEMORY MAP — GR740 Addresses
 * ====================================================================== */
#define BOOT_PROM_BASE          0x00000000U
#define BOOT_PROM_SIZE          0x00040000U  /**< 256 KB                  */

#define SRAM_BASE               0x40000000U
#define SRAM_SIZE               0x02000000U  /**< 32 MB                   */

#define MRAM_BASE               0x20000000U
#define MRAM_SIZE               0x00800000U  /**< 8 MB                    */

#define EEPROM_BASE             0x30000000U
#define EEPROM_SIZE             0x00400000U  /**< 4 MB                    */

/* ======================================================================
 * INTERRUPT NUMBERS (GR740 IRQ assignments)
 * ====================================================================== */
#define IRQ_UART0               2U
#define IRQ_UART1               3U
#define IRQ_TIMER0              8U
#define IRQ_TIMER1              9U
#define IRQ_GRCAN0              10U
#define IRQ_GRCAN1              11U
#define IRQ_GRSPW0              12U
#define IRQ_GRSPW1              13U
#define IRQ_GRSPW2              14U
#define IRQ_GRSPW3              15U
#define IRQ_SPICTRL0            16U
#define IRQ_I2CMST0             17U
#define IRQ_GPIO0               18U

/* ======================================================================
 * REGISTER ACCESS MACROS
 * ====================================================================== */

/**
 * @brief Read a 32-bit hardware register.
 * @param base Base address of the peripheral.
 * @param offset Register offset from base.
 * @return Register value.
 */
#define REG_READ(base, offset) \
    (*(volatile uint32_t *)((uint32_t)(base) + (uint32_t)(offset)))

/**
 * @brief Write a 32-bit hardware register.
 * @param base Base address of the peripheral.
 * @param offset Register offset from base.
 * @param value Value to write.
 */
#define REG_WRITE(base, offset, value) \
    (*(volatile uint32_t *)((uint32_t)(base) + (uint32_t)(offset)) = (uint32_t)(value))

/**
 * @brief Set bits in a 32-bit hardware register.
 * @param base Base address of the peripheral.
 * @param offset Register offset from base.
 * @param mask Bits to set.
 */
#define REG_SET(base, offset, mask) \
    REG_WRITE((base), (offset), REG_READ((base), (offset)) | (uint32_t)(mask))

/**
 * @brief Clear bits in a 32-bit hardware register.
 * @param base Base address of the peripheral.
 * @param offset Register offset from base.
 * @param mask Bits to clear.
 */
#define REG_CLR(base, offset, mask) \
    REG_WRITE((base), (offset), REG_READ((base), (offset)) & ~(uint32_t)(mask))

#endif /* HW_CONFIG_H */
