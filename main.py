import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Define as sementes aleatórias para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Nomes das classes usados nos relatórios e nos gráficos
NOMES_CLASSES = ['Normal', 'Anomalia']


class PetroleumDataGenerator:
    """Gera dados sintéticos de engenharia de petróleo com anomalias"""

    def __init__(self, n_samples=10000, anomaly_fraction=0.05):
        self.n_samples = n_samples
        self.anomaly_fraction = anomaly_fraction
        self.n_anomalies = int(n_samples * anomaly_fraction)
        self.n_normal = n_samples - self.n_anomalies

    def generate_data(self):
        """Gera dados sintéticos de engenharia de petróleo"""

        # Condições normais de operação
        # Pressão (psi) - faixa típica de 2000-8000 psi
        pressure = np.random.normal(5000, 800, self.n_normal)
        pressure = np.clip(pressure, 2000, 8000)

        # Temperatura (°F) - faixa típica de 150-300°F
        temperature = np.random.normal(200, 30, self.n_normal)
        temperature = np.clip(temperature, 150, 300)

        # Vazão (bbl/dia) - correlacionada com a pressão
        flow_rate = 50 + 0.8 * pressure + np.random.normal(0, 200, self.n_normal)
        flow_rate = np.clip(flow_rate, 100, 8000)

        # Razão Gás-Óleo (scf/bbl) - faixa típica de 500-2000
        gor = np.random.normal(1000, 200, self.n_normal)
        gor = np.clip(gor, 500, 2000)

        # Corte de água (%) - porcentagem de água na produção
        water_cut = np.random.beta(2, 5, self.n_normal) * 100
        water_cut = np.clip(water_cut, 0, 80)

        # Pressão de cabeça de poço (psi) - correlacionada com a pressão do reservatório
        wellhead_pressure = pressure * 0.7 + np.random.normal(0, 100, self.n_normal)
        wellhead_pressure = np.clip(wellhead_pressure, 1000, 6000)

        # Tamanho do choke (64avos de polegada) - afeta a vazão
        choke_size = np.random.choice([16, 20, 24, 32, 40, 48, 56, 64], self.n_normal)

        # Eficiência da bomba (%) - afeta a produção
        pump_efficiency = np.random.normal(85, 10, self.n_normal)
        pump_efficiency = np.clip(pump_efficiency, 60, 100)

        # Geração das anomalias
        # Anomalia 1: Falha de equipamento (baixa eficiência da bomba, pressão anormal)
        anom1_count = self.n_anomalies // 3
        anom1_pressure = np.random.normal(3000, 500, anom1_count)
        anom1_temperature = np.random.normal(180, 20, anom1_count)
        anom1_flow_rate = np.random.normal(1000, 300, anom1_count)
        anom1_gor = np.random.normal(1500, 300, anom1_count)
        anom1_water_cut = np.random.normal(60, 15, anom1_count)
        anom1_wellhead_pressure = np.random.normal(2000, 300, anom1_count)
        anom1_choke_size = np.random.choice([16, 20, 24], anom1_count)
        anom1_pump_efficiency = np.random.normal(40, 10, anom1_count)

        # Anomalia 2: Problemas de reservatório (alto corte de água, GOR variando)
        anom2_count = self.n_anomalies // 3
        anom2_pressure = np.random.normal(4500, 600, anom2_count)
        anom2_temperature = np.random.normal(220, 25, anom2_count)
        anom2_flow_rate = np.random.normal(3000, 800, anom2_count)
        anom2_gor = np.random.normal(2500, 400, anom2_count)
        anom2_water_cut = np.random.normal(90, 5, anom2_count)
        anom2_wellhead_pressure = np.random.normal(3500, 400, anom2_count)
        anom2_choke_size = np.random.choice([32, 40, 48], anom2_count)
        anom2_pump_efficiency = np.random.normal(75, 12, anom2_count)

        # Anomalia 3: Condições extremas de operação
        anom3_count = self.n_anomalies - anom1_count - anom2_count
        anom3_pressure = np.random.normal(8500, 500, anom3_count)
        anom3_temperature = np.random.normal(350, 30, anom3_count)
        anom3_flow_rate = np.random.normal(9000, 1000, anom3_count)
        anom3_gor = np.random.normal(3000, 500, anom3_count)
        anom3_water_cut = np.random.normal(95, 3, anom3_count)
        anom3_wellhead_pressure = np.random.normal(7000, 300, anom3_count)
        anom3_choke_size = np.random.choice([56, 64], anom3_count)
        anom3_pump_efficiency = np.random.normal(95, 5, anom3_count)

        # Combina os dados normais e anômalos
        all_pressure = np.concatenate([pressure, anom1_pressure, anom2_pressure, anom3_pressure])
        all_temperature = np.concatenate([temperature, anom1_temperature, anom2_temperature, anom3_temperature])
        all_flow_rate = np.concatenate([flow_rate, anom1_flow_rate, anom2_flow_rate, anom3_flow_rate])
        all_gor = np.concatenate([gor, anom1_gor, anom2_gor, anom3_gor])
        all_water_cut = np.concatenate([water_cut, anom1_water_cut, anom2_water_cut, anom3_water_cut])
        all_wellhead_pressure = np.concatenate([wellhead_pressure, anom1_wellhead_pressure, anom2_wellhead_pressure, anom3_wellhead_pressure])
        all_choke_size = np.concatenate([choke_size, anom1_choke_size, anom2_choke_size, anom3_choke_size])
        all_pump_efficiency = np.concatenate([pump_efficiency, anom1_pump_efficiency, anom2_pump_efficiency, anom3_pump_efficiency])

        # Cria os rótulos (0 para normal, 1 para anomalia)
        labels = np.concatenate([np.zeros(self.n_normal), np.ones(self.n_anomalies)])

        # Cria o DataFrame
        data = pd.DataFrame({
            'pressure_psi': all_pressure,
            'temperature_f': all_temperature,
            'flow_rate_bbl_day': all_flow_rate,
            'gor_scf_bbl': all_gor,
            'water_cut_percent': all_water_cut,
            'wellhead_pressure_psi': all_wellhead_pressure,
            'choke_size_64ths': all_choke_size,
            'pump_efficiency_percent': all_pump_efficiency,
            'is_anomaly': labels
        })

        # Embaralha os dados
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        return data


class AutoEncoder:
    """Autoencoder para detecção de anomalias"""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, encoding_dim=4):
        """Constrói o modelo autoencoder"""
        input_layer = Input(shape=(self.input_dim,))

        # Codificador (Encoder)
        encoded = Dense(16, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)

        # Decodificador (Decoder)
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # Modelo autoencoder
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='mse',
                           metrics=['mae'])

        return self.model

    def train(self, X_train, epochs=100, batch_size=32, validation_split=0.2):
        """Treina o autoencoder"""
        # Padroniza os dados
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Treina apenas com dados normais (assumindo que a maioria dos dados é normal)
        history = self.model.fit(X_train_scaled, X_train_scaled,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 shuffle=True,
                                 verbose=1)

        return history

    def predict_anomalies(self, X_test, threshold_percentile=95):
        """Prevê anomalias com base no erro de reconstrução"""
        X_test_scaled = self.scaler.transform(X_test)

        # Obtém as reconstruções
        reconstructions = self.model.predict(X_test_scaled)

        # Calcula o erro de reconstrução
        mse = np.mean(np.square(X_test_scaled - reconstructions), axis=1)

        # Define o limiar com base em um percentil
        threshold = np.percentile(mse, threshold_percentile)

        # Prevê as anomalias
        anomalies = mse > threshold

        return anomalies, mse, threshold


class PetroleumAnomalyDetector:
    """Classe principal para detecção de anomalias em engenharia de petróleo"""

    def __init__(self):
        self.data = None
        self.autoencoder = None
        self.isolation_forest = None
        self.scaler = StandardScaler()

    def load_data(self, data):
        """Carrega os dados de engenharia de petróleo"""
        self.data = data

    def preprocess_data(self):
        """Pré-processa os dados para aprendizado de máquina"""
        # Separa as variáveis (features) dos rótulos
        X = self.data.drop('is_anomaly', axis=1)
        y = self.data['is_anomaly']

        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train_isolation_forest(self, X_train, contamination=0.05):
        """Treina o modelo Isolation Forest"""
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        # Padroniza os dados
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Treina o modelo
        self.isolation_forest.fit(X_train_scaled)

        return self.isolation_forest

    def train_autoencoder(self, X_train):
        """Treina o modelo AutoEncoder"""
        # Inicializa o autoencoder
        self.autoencoder = AutoEncoder(input_dim=X_train.shape[1])
        self.autoencoder.build_model()

        # Treina o modelo
        history = self.autoencoder.train(X_train, epochs=50, batch_size=32)

        return history

    def evaluate_models(self, X_test, y_test):
        """Avalia os dois modelos"""
        results = {}

        # Previsões do Isolation Forest
        X_test_scaled = self.scaler.transform(X_test)
        if_predictions = self.isolation_forest.predict(X_test_scaled)
        if_predictions = (if_predictions == -1).astype(int)  # Converte para 0/1

        # Previsões do AutoEncoder
        ae_predictions, mse, threshold = self.autoencoder.predict_anomalies(X_test)
        ae_predictions = ae_predictions.astype(int)

        # Armazena os resultados
        results['isolation_forest'] = {
            'predictions': if_predictions,
            'classification_report': classification_report(
                y_test, if_predictions, target_names=NOMES_CLASSES),
            'confusion_matrix': confusion_matrix(y_test, if_predictions)
        }

        results['autoencoder'] = {
            'predictions': ae_predictions,
            'reconstruction_errors': mse,
            'threshold': threshold,
            'classification_report': classification_report(
                y_test, ae_predictions, target_names=NOMES_CLASSES),
            'confusion_matrix': confusion_matrix(y_test, ae_predictions)
        }

        return results

    def plot_results(self, X_test, y_test, results):
        """Plota os resultados"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Gráfico 1: Distribuição dos dados
        axes[0, 0].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'],
                           c=y_test, alpha=0.6, cmap='viridis')
        axes[0, 0].set_xlabel('Pressão (psi)')
        axes[0, 0].set_ylabel('Vazão (bbl/dia)')
        axes[0, 0].set_title('Anomalias Reais')
        axes[0, 0].grid(True, alpha=0.3)

        # Gráfico 2: Resultados do Isolation Forest
        axes[0, 1].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'],
                           c=results['isolation_forest']['predictions'], alpha=0.6, cmap='viridis')
        axes[0, 1].set_xlabel('Pressão (psi)')
        axes[0, 1].set_ylabel('Vazão (bbl/dia)')
        axes[0, 1].set_title('Previsões do Isolation Forest')
        axes[0, 1].grid(True, alpha=0.3)

        # Gráfico 3: Resultados do AutoEncoder
        axes[0, 2].scatter(X_test['pressure_psi'], X_test['flow_rate_bbl_day'],
                           c=results['autoencoder']['predictions'], alpha=0.6, cmap='viridis')
        axes[0, 2].set_xlabel('Pressão (psi)')
        axes[0, 2].set_ylabel('Vazão (bbl/dia)')
        axes[0, 2].set_title('Previsões do AutoEncoder')
        axes[0, 2].grid(True, alpha=0.3)

        # Gráfico 4: Erros de reconstrução
        axes[1, 0].hist(results['autoencoder']['reconstruction_errors'], bins=50, alpha=0.7)
        axes[1, 0].axvline(results['autoencoder']['threshold'], color='red', linestyle='--',
                           label=f'Limiar: {results["autoencoder"]["threshold"]:.4f}')
        axes[1, 0].set_xlabel('Erro de Reconstrução')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].set_title('Erros de Reconstrução do AutoEncoder')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Gráfico 5: Matriz de Confusão - Isolation Forest
        cm_if = results['isolation_forest']['confusion_matrix']
        sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                    xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES)
        axes[1, 1].set_title('Matriz de Confusão - Isolation Forest')
        axes[1, 1].set_xlabel('Previsto')
        axes[1, 1].set_ylabel('Real')

        # Gráfico 6: Matriz de Confusão - AutoEncoder
        cm_ae = results['autoencoder']['confusion_matrix']
        sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
                    xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES)
        axes[1, 2].set_title('Matriz de Confusão - AutoEncoder')
        axes[1, 2].set_xlabel('Previsto')
        axes[1, 2].set_ylabel('Real')

        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self):
        """Executa a análise completa de detecção de anomalias"""
        print("🛢️  PROJETO DE DETECÇÃO DE ANOMALIAS EM ENGENHARIA DE PETRÓLEO")
        print("=" * 60)

        # Gera os dados
        print("\n1. Gerando dados sintéticos de engenharia de petróleo...")
        data_generator = PetroleumDataGenerator(n_samples=5000, anomaly_fraction=0.08)
        self.data = data_generator.generate_data()

        print(f"   Foram geradas {len(self.data)} amostras com {int(self.data['is_anomaly'].sum())} anomalias")
        print(f"   Variáveis: {list(self.data.columns[:-1])}")

        # Exibe estatísticas básicas
        print("\n2. Visão Geral dos Dados:")
        print(self.data.describe())

        # Pré-processa os dados
        print("\n3. Pré-processando os dados...")
        X_train, X_test, y_train, y_test = self.preprocess_data()
        print(f"   Amostras de treino: {len(X_train)}")
        print(f"   Amostras de teste: {len(X_test)}")

        # Treina os modelos
        print("\n4. Treinando o Isolation Forest...")
        self.train_isolation_forest(X_train, contamination=0.08)
        print("   ✓ Isolation Forest treinado com sucesso")

        print("\n5. Treinando o AutoEncoder...")
        history = self.train_autoencoder(X_train)
        print("   ✓ AutoEncoder treinado com sucesso")

        # Avalia os modelos
        print("\n6. Avaliando os modelos...")
        results = self.evaluate_models(X_test, y_test)

        # Exibe os resultados
        print("\n7. RESULTADOS:")
        print("\n" + "=" * 40)
        print("RESULTADOS DO ISOLATION FOREST:")
        print("=" * 40)
        print(results['isolation_forest']['classification_report'])

        print("\n" + "=" * 40)
        print("RESULTADOS DO AUTOENCODER:")
        print("=" * 40)
        print(results['autoencoder']['classification_report'])

        # Plota os resultados
        print("\n8. Gerando visualizações...")
        self.plot_results(X_test, y_test, results)

        # Análise das variáveis
        print("\n9. Análise das Variáveis:")
        feature_names = X_train.columns
        print(f"   Variáveis analisadas: {list(feature_names)}")

        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO:")
        print("=" * 60)
        print("Os dois modelos foram treinados e avaliados com sucesso para")
        print("a detecção de anomalias em engenharia de petróleo. A análise inclui:")
        print("• Detecção de falhas de equipamento")
        print("• Monitoramento do desempenho do reservatório")
        print("• Identificação de condições extremas de operação")
        print("• Insights para otimização da produção")

        return results


# Executa a análise completa
if __name__ == "__main__":
    detector = PetroleumAnomalyDetector()
    results = detector.run_complete_analysis()