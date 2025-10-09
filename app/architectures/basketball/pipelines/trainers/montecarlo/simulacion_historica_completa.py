"""
SIMULACIÓN HISTÓRICA COMPLETA NBA
=================================

Pipeline que procesa TODAS las fechas disponibles en el dataset
para entrenar el sistema Montecarlo progresivamente.
"""

import sys
import os
# Agregar el directorio raíz al PYTHONPATH para resolver las importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.montecarlo.enhanced_engine import EnhancedMonteCarloEngine
from app.architectures.basketball.src.models.montecarlo.simulator import NBAGameSimulator
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
import time

class SimulacionHistoricaCompleta:
    """Pipeline para simulación histórica completa con aprendizaje progresivo"""
    
    def __init__(self):
        self.setup_logging()
        
        # Cargar datos
        self.logger.info("Cargando datasets completos...")
        loader = NBADataLoader('app/architectures/basketball/data/players.csv', 'app/architectures/basketball/data/height.csv', 'app/architectures/basketball/data/teams.csv')
        self.players_df, self.teams_df = loader.load_data()
        
        # Preparar fechas
        self.players_df['Date'] = pd.to_datetime(self.players_df['Date'])
        self.teams_df['Date'] = pd.to_datetime(self.teams_df['Date'])
        
        # Obtener fechas únicas ordenadas
        self.fechas_disponibles = sorted(self.players_df['Date'].unique())
        
        # Inicializar sistema mejorado
        self.enhanced_engine = EnhancedMonteCarloEngine(self.players_df, self.teams_df)
        self.simulator = NBAGameSimulator(self.enhanced_engine, num_simulations=10000)
        
        # Métricas de aprendizaje
        self.metricas_aprendizaje = {
            'total_simulaciones': 0,
            'aciertos_progresivos': [],
            'precision_por_temporada': {}
        }
        
        self.logger.info(f"Sistema inicializado con {len(self.fechas_disponibles)} fechas disponibles")

    def setup_logging(self):
        """Configurar logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(__name__)

    def procesar_simulacion_historica(self, modo_rapido=False, saltar_cada=1):
        """Procesa simulación histórica completa"""
        
        fechas_a_procesar = self.fechas_disponibles[::saltar_cada]
        
        self.logger.info(f"INICIANDO SIMULACIÓN HISTÓRICA")
        self.logger.info(f"Fechas a procesar: {len(fechas_a_procesar)} de {len(self.fechas_disponibles)}")
        
        if modo_rapido:
            self.simulator.num_simulations = 5000
        
        total_partidos = 0
        tiempo_inicio = time.time()
        
        for i, fecha in enumerate(fechas_a_procesar):
            try:
                fecha_str = fecha.strftime('%Y-%m-%d')
                self.logger.info(f"[{i+1}/{len(fechas_a_procesar)}] Procesando {fecha_str}")
                
                # Simular fecha
                partidos_simulados = self.simular_fecha(fecha)
                total_partidos += partidos_simulados
                
                # Guardar progreso cada 20 fechas
                if (i + 1) % 20 == 0:
                    self.guardar_progreso(i + 1, len(fechas_a_procesar))
                
            except Exception as e:
                self.logger.error(f"Error procesando {fecha_str}: {str(e)}")
                continue
        
        # Reporte final
        tiempo_total = time.time() - tiempo_inicio
        self.generar_reporte_final(len(fechas_a_procesar), total_partidos, tiempo_total)

    def simular_fecha(self, fecha):
        """Simula todos los partidos de una fecha"""
        
        fecha_str = fecha.strftime('%m/%d/%Y')
        partidos = self.obtener_partidos_fecha(fecha)
        
        partidos_simulados = 0
        
        for partido in partidos:
            try:
                resultado = self.simulator.simulate_game(
                    home_team=partido['home_team'],
                    away_team=partido['away_team'],
                    date=fecha_str
                )
                
                if resultado:
                    partidos_simulados += 1
                    # Aplicar aprendizaje si hay datos reales
                    self.aplicar_aprendizaje(resultado, fecha, partido)
                    
            except Exception as e:
                self.logger.warning(f"Error simulando {partido['away_team']} @ {partido['home_team']}")
                continue
        
        return partidos_simulados

    def obtener_partidos_fecha(self, fecha):
        """Obtiene partidos únicos de una fecha"""
        
        fecha_data = self.teams_df[self.teams_df['Date'] == fecha]
        
        if fecha_data.empty:
            return []
        
        partidos = []
        partidos_procesados = set()
        
        for _, row in fecha_data.iterrows():
            team = row['Team']
            opp = row['Opp']
            is_home = row.get('is_home', 1) == 1
            
            if is_home:
                home_team = team
                away_team = opp
            else:
                home_team = opp  
                away_team = team
            
            partido_key = f"{away_team}@{home_team}"
            if partido_key not in partidos_procesados:
                partidos.append({
                    'home_team': home_team,
                    'away_team': away_team
                })
                partidos_procesados.add(partido_key)
        
        return partidos

    def aplicar_aprendizaje(self, resultado_simulacion, fecha, partido):
        """Aplica aprendizaje comparando con datos reales"""
        
        try:
            resultado_real = self.buscar_resultado_real(
                partido['home_team'], 
                partido['away_team'], 
                fecha
            )
            
            if resultado_real:
                prediccion_correcta = self.validar_prediccion(resultado_simulacion, resultado_real)
                
                # Actualizar métricas
                self.metricas_aprendizaje['total_simulaciones'] += 1
                self.metricas_aprendizaje['aciertos_progresivos'].append(prediccion_correcta)
                
                # Actualizar sistema de aprendizaje del motor
                if hasattr(self.enhanced_engine, 'update_prediction_accuracy'):
                    self.enhanced_engine.update_prediction_accuracy(
                        resultado_simulacion, 
                        resultado_real
                    )
                    
        except Exception:
            pass

    def buscar_resultado_real(self, home_team, away_team, fecha):
        """Busca resultado real del partido"""
        
        try:
            home_data = self.teams_df[
                (self.teams_df['Date'] == fecha) & 
                (self.teams_df['Team'] == home_team) &
                (self.teams_df['Opp'] == away_team)
            ]
            
            away_data = self.teams_df[
                (self.teams_df['Date'] == fecha) & 
                (self.teams_df['Team'] == away_team) &
                (self.teams_df['Opp'] == home_team)
            ]
            
            if not home_data.empty and not away_data.empty:
                home_score = home_data.iloc[0]['PTS']
                away_score = away_data.iloc[0]['PTS']
                
                return {
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': 'home' if home_score > away_score else 'away',
                    'margin': abs(home_score - away_score)
                }
            
            return None
            
        except Exception:
            return None

    def validar_prediccion(self, simulacion, resultado_real):
        """Valida si la predicción fue correcta"""
        
        try:
            win_probs = simulacion.get('win_probabilities', {})
            prediccion_ganador = 'home' if win_probs.get('home_win', 0) > 0.5 else 'away'
            return prediccion_ganador == resultado_real['winner']
        except Exception:
            return False

    def guardar_progreso(self, fechas_procesadas, total_fechas):
        """Guarda progreso del aprendizaje"""
        
        os.makedirs('cache/simulacion_historica', exist_ok=True)
        
        progreso = {
            'timestamp': datetime.now().isoformat(),
            'fechas_procesadas': fechas_procesadas,
            'total_fechas': total_fechas,
            'progreso_porcentaje': (fechas_procesadas / total_fechas) * 100,
            'metricas': self.metricas_aprendizaje.copy()
        }
        
        with open('cache/simulacion_historica/progreso.json', 'w') as f:
            json.dump(progreso, f, indent=2, default=str)
        
        # Mostrar precisión actual
        if self.metricas_aprendizaje['aciertos_progresivos']:
            aciertos = sum(self.metricas_aprendizaje['aciertos_progresivos'])
            total = len(self.metricas_aprendizaje['aciertos_progresivos'])
            precision = aciertos / total
            self.logger.info(f"Precisión actual: {precision:.1%} ({aciertos}/{total})")

    def generar_reporte_final(self, fechas_procesadas, partidos_simulados, tiempo_total):
        """Genera reporte final"""
        
        self.logger.info(f"\nSIMULACIÓN HISTÓRICA COMPLETADA")
        self.logger.info(f"Fechas procesadas: {fechas_procesadas}")
        self.logger.info(f"Partidos simulados: {partidos_simulados}")
        self.logger.info(f"Tiempo total: {tiempo_total/60:.1f} minutos")
        
        if self.metricas_aprendizaje['aciertos_progresivos']:
            aciertos = sum(self.metricas_aprendizaje['aciertos_progresivos'])
            total = len(self.metricas_aprendizaje['aciertos_progresivos'])
            precision_final = aciertos / total
            self.logger.info(f"Precisión final: {precision_final:.1%}")
        
        # Guardar reporte final
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'fechas_procesadas': fechas_procesadas,
            'partidos_simulados': partidos_simulados,
            'tiempo_total_minutos': tiempo_total / 60,
            'precision_final': precision_final if 'precision_final' in locals() else 0,
            'metricas_completas': self.metricas_aprendizaje
        }
        
        with open('cache/simulacion_historica/reporte_final.json', 'w') as f:
            json.dump(reporte, f, indent=2, default=str)
        
        self.logger.info("Sistema entrenado y listo para predicciones mejoradas")

def main():
    """Función principal"""
    
    print("SIMULACIÓN HISTÓRICA COMPLETA NBA")
    print("=" * 40)
    print("1. Simulación completa (todas las fechas)")
    print("2. Simulación rápida (cada 2 fechas)")  
    print("3. Simulación muestra (cada 5 fechas)")
    print("4. Solo últimos 60 días")
    
    try:
        opcion = input("Seleccione opción (1-4): ").strip()
        
        simulador = SimulacionHistoricaCompleta()
        
        if opcion == "1":
            simulador.procesar_simulacion_historica(modo_rapido=False, saltar_cada=1)
        elif opcion == "2":
            simulador.procesar_simulacion_historica(modo_rapido=True, saltar_cada=2)
        elif opcion == "3":
            simulador.procesar_simulacion_historica(modo_rapido=True, saltar_cada=5)
        elif opcion == "4":
            # Solo últimos 60 días
            fecha_fin = simulador.fechas_disponibles[-1]
            fecha_inicio = fecha_fin - timedelta(days=60)
            fechas_filtradas = [f for f in simulador.fechas_disponibles 
                              if f >= fecha_inicio]
            simulador.fechas_disponibles = fechas_filtradas
            simulador.procesar_simulacion_historica(modo_rapido=False, saltar_cada=1)
        else:
            print("Opción no válida")
            
    except KeyboardInterrupt:
        print("\nSimulación interrumpida")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 