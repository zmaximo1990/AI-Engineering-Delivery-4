# Catálogo de Códigos de Error y Alarmas

NimbusOS 4.x - Gateway Nimbus Edge GX-200 / GX-200L

**Código:** NE-300 **Versión:** 4.2 **Fecha:** 2026-09-14

**Público objetivo:** Soporte técnico, operadores y mantenimiento

## 1. Cómo leer un código

Los códigos de error de Nimbus Edge tienen el formato **NE-E**`<categoría><número>`: la primera cifra indica la categoría y las tres siguientes identifican el evento. Cada código tiene una severidad que determina cómo se refleja en los LED y en las alarmas de Nimbus Cloud.

| **Rango** | **Categoría** |
| --- | --- |
| NE-E1xxx | Conectividad con la nube y seguridad (TLS, credenciales) |
| NE-E2xxx | Hardware y ejecución (Modbus, watchdog, temperatura) |
| NE-E3xxx | Almacenamiento y firmware |
| NE-E4xxx | Red local, WAN y celular |
| NE-E5xxx | Sistema y configuración |

| **Severidad** | **Indicación en LED ERR** | **Alarma en la nube** |
| --- | --- | --- |
| Información | Apagado | No genera alarma |
| Advertencia | Rojo parpadeante lento | Alarma de severidad `warning` |
| Crítico | Rojo fijo | Alarma de severidad `critical` |

## 2. Tabla resumen

| **Código** | **Severidad** | **Título** |
| --- | --- | --- |
| NE-E1001 | Crítico | Fallo de autenticación TLS con el broker |
| NE-E1002 | Advertencia | Broker MQTT inalcanzable |
| NE-E1017 | Crítico | Credenciales de dispositivo rechazadas |
| NE-E1042 | Advertencia | Buffer de telemetría lleno |
| NE-E2003 | Advertencia | Tiempo de espera agotado en dispositivo Modbus |
| NE-E2007 | Crítico | Reinicio por watchdog |
| NE-E2015 | Crítico | Sobretemperatura del procesador |
| NE-E3001 | Advertencia | Espacio de almacenamiento bajo |
| NE-E3010 | Crítico | Falla en la verificación de firma del firmware |
| NE-E3011 | Advertencia | Actualización interrumpida con rollback automático |
| NE-E4004 | Advertencia | Pérdida de enlace WAN y conmutación a LTE |
| NE-E4009 | Crítico | SIM no detectada o PIN incorrecto |
| NE-E5001 | Advertencia | Deriva de reloj excesiva |
| NE-E5020 | Advertencia | Configuración inválida: parámetro fuera de rango |

## 3. Detalle por código

### NE-E1001 - Fallo de autenticación TLS con el broker

**Severidad:** crítico. El gateway no pudo validar el certificado del broker MQTT o el broker rechazó el certificado del dispositivo.

**Causas probables:** reloj del equipo desfasado, certificado de dispositivo vencido, CA raíz no confiable por inspección TLS en un firewall corporativo.

**Acción recomendada:** verifique la hora con `nbctl time sync`; revise la vigencia con `nbctl cert status`; si el certificado venció o está por vencer, ejecute `nbctl cert renew`. Si hay un firewall con inspección TLS, excluya el dominio del broker de la inspección.

### NE-E1002 - Broker MQTT inalcanzable

**Severidad:** advertencia. Se agotó el tiempo de conexión con el broker en el puerto **8883**.

**Causas probables:** DNS que no resuelve, puerto 8883 bloqueado por firewall, caída de la WAN.

**Acción recomendada:** ejecute `nbctl net test` para comprobar DNS y conectividad al puerto 8883. El gateway reintenta automáticamente con espera exponencial y mientras tanto acumula la telemetría en el buffer local.

### NE-E1017 - Credenciales de dispositivo rechazadas

**Severidad:** crítico. Nimbus Cloud rechazó el token de aprovisionamiento o la identidad del dispositivo.

**Causas probables:** el token de aprovisionamiento expiró (vigencia de 72 horas), ya fue usado, o el dispositivo fue revocado tras un restablecimiento de fábrica.

**Acción recomendada:** genere un token nuevo en Nimbus Console y vuelva a aprovisionar el equipo desde el asistente web.

### NE-E1042 - Buffer de telemetría lleno

**Severidad:** advertencia. El buffer local en la eMMC alcanzó el límite `telemetry.buffer_max_mb` (por defecto 256 MB) y se están aplicando las reglas de `telemetry.buffer_policy`.

**Causas probables:** la conexión con la nube estuvo caída por un período prolongado o la tasa de muestreo es demasiado alta para el ancho de banda disponible.

**Acción recomendada:** restablezca la conectividad; aumente `telemetry.sample_interval_ms` o `telemetry.batch_size`; revise la política. Con `drop_oldest` se descartan los datos más antiguos, con `drop_newest` se descartan los nuevos y con `block` se detiene el muestreo.

### NE-E2003 - Tiempo de espera agotado en dispositivo Modbus

**Severidad:** advertencia. Un esclavo Modbus no respondió dentro de `modbus.timeout_ms` (por defecto 1500 ms) tras agotar `modbus.retries` (por defecto 3).

**Causas probables:** cableado A/B invertido, falta de resistencia de terminación de 120 ohmios, velocidad o paridad distintas entre el gateway y el esclavo, identificador de esclavo incorrecto.

**Acción recomendada:** verifique el cableado y la terminación; ejecute `nbctl modbus scan --port rs485 --baud 9600` para detectar los esclavos presentes y confirmar sus parámetros.

### NE-E2007 - Reinicio por watchdog

**Severidad:** crítico. El proceso `nimbusd` dejó de responder durante `watchdog.timeout_s` (por defecto 60 segundos) y el sistema se reinició automáticamente.

**Causas probables:** en versiones anteriores a 4.2.1, un sondeo Modbus con `modbus.poll_interval_ms` menor a 200 y más de 64 registros configurados podía saturar el proceso. También puede originarse por caídas de tensión de alimentación por debajo de 9 V o por memoria insuficiente.

**Acción recomendada:** actualice a NimbusOS 4.2.1 o superior; si no es posible, aumente `modbus.poll_interval_ms` a 200 o más. Revise la fuente de alimentación y recolecte un paquete de diagnóstico con `nbctl diag --full`.

### NE-E2015 - Sobretemperatura del procesador

**Severidad:** crítico. La temperatura del procesador superó los **85 °C**, momento en que el equipo reduce la frecuencia (throttling). Si llega a **95 °C**, el equipo se apaga para protegerse.

**Causas probables:** ventilación insuficiente en el tablero, distancia menor a 30 mm con otros equipos, temperatura ambiente por encima de 70 °C.

**Acción recomendada:** mejore la ventilación o agregue extracción forzada en el gabinete. Monitoree la métrica `cpu_temp_c` desde la API.

### NE-E3001 - Espacio de almacenamiento bajo

**Severidad:** advertencia. Queda menos del **10 %** de espacio libre en la eMMC.

**Acción recomendada:** reduzca `logging.max_files`, exporte y elimine diagnósticos antiguos en `/var/tmp`, o reduzca `telemetry.buffer_max_mb`. Con menos de 500 MB libres no es posible instalar actualizaciones de firmware.

### NE-E3010 - Falla en la verificación de firma del firmware

**Severidad:** crítico. El paquete `.nbfw` no pasó la verificación de firma Ed25519 o de suma SHA-256 y fue descartado antes de escribirse.

**Causas probables:** descarga corrupta o paquete modificado.

**Acción recomendada:** descargue nuevamente el paquete desde Nimbus Console y verifique su suma SHA-256 publicada. La partición activa no se modifica.

### NE-E3011 - Actualización interrumpida con rollback automático

**Severidad:** advertencia. La actualización de firmware no superó la verificación posterior de 300 segundos y el equipo volvió a la partición anterior.

**Acción recomendada:** consulte `nbctl ota status` para ver el motivo, corrija la causa (por ejemplo, falta de espacio o conectividad) y reintente. Ver documento NE-410.

### NE-E4004 - Pérdida de enlace WAN y conmutación a LTE

**Severidad:** advertencia. Solo aplica al GX-200L. El enlace WAN por Ethernet estuvo caído durante `network.wan.failover_timeout_s` (por defecto 45 segundos) y el tráfico se desvió al módem LTE.

**Acción recomendada:** restablezca el enlace por Ethernet. El equipo vuelve automáticamente a WAN cableada cuando el enlace se mantiene estable durante 120 segundos.

### NE-E4009 - SIM no detectada o PIN incorrecto

**Severidad:** crítico. El módem LTE no detecta la nano-SIM o el PIN configurado es incorrecto.

**Acción recomendada:** apague el equipo, verifique que la SIM esté bien insertada y con el chip hacia abajo, y confirme el PIN. Después de 3 intentos con PIN incorrecto la SIM se bloquea y requiere el código PUK del operador.

### NE-E5001 - Deriva de reloj excesiva

**Severidad:** advertencia. La diferencia entre la hora del equipo y la de referencia NTP supera los **5 segundos**.

**Acción recomendada:** verifique que el puerto UDP 123 esté permitido hacia los servidores NTP y ejecute `nbctl time sync`. Un reloj desfasado puede provocar el código NE-E1001.

### NE-E5020 - Configuración inválida: parámetro fuera de rango

**Severidad:** advertencia. `nbctl config validate` detectó un valor fuera del rango admitido, por ejemplo `mqtt.keepalive_interval` menor a 10. El equipo conserva la última configuración válida.

**Acción recomendada:** corrija el valor según los rangos documentados en el documento NE-220 y vuelva a aplicar la configuración.

## 4. Cómo consultar los códigos activos

- **Localmente:** `nbctl status` lista los códigos activos con su hora de inicio.
- **En el registro:** los códigos se escriben en `/var/log/nimbus/nimbusd.log` con el prefijo `NE-E`.
- **En la nube:** `GET /v3/alarms?severity=critical` devuelve las alarmas críticas (ver documento NE-210).
