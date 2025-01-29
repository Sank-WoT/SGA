import numpy as np

class SAGLinearRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-4):
        """
        Инициализация параметров:
        - learning_rate: скорость обучения
        - max_iters: максимальное количество итераций
        - tol: критерий остановки (разница между потерями на двух последовательных итерациях)
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Обучение модели с использованием SAG.
        - X: матрица признаков (n_samples, n_features)
        - y: целевые значения (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Инициализация весов и смещения
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Инициализация градиентов для каждого примера
        gradients = np.zeros((n_samples, n_features))
        gradient_bias = np.zeros(n_samples)
        
        # История потерь для критерия остановки
        prev_loss = float('inf')
        
        for iter in range(self.max_iters):
            # Случайный выбор индекса примера
            i = np.random.randint(0, n_samples)
            
            # Вычисление предсказания
            prediction = np.dot(X[i], self.weights) + self.bias
            
            # Вычисление градиента для выбранного примера
            error = prediction - y[i]
            new_gradient = X[i] * error
            new_gradient_bias = error
            
            # Обновление массива градиентов
            gradients[i] = new_gradient
            gradient_bias[i] = new_gradient_bias
            
            # Пересчет среднего градиента
            avg_gradient = np.mean(gradients, axis=0)
            avg_gradient_bias = np.mean(gradient_bias)
            
            # Обновление весов и смещения
            self.weights -= self.learning_rate * avg_gradient
            self.bias -= self.learning_rate * avg_gradient_bias
            
            # Вычисление потерь для критерия остановки
            if iter % 100 == 0:
                current_loss = self._compute_loss(X, y)
                if abs(prev_loss - current_loss) < self.tol:
                    print(f"Ранняя остановка на итерации {iter}")
                    break
                prev_loss = current_loss

    def predict(self, X):
        """
        Предсказание целевых значений.
        - X: матрица признаков (n_samples, n_features)
        """
        return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, X, y):
        """
        Вычисление функции потерь (MSE).
        - X: матрица признаков (n_samples, n_features)
        - y: целевые значения (n_samples,)
        """
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
        
        
  # Генерация синтетических данных
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Обучение модели
sag_model = SAGLinearRegression(learning_rate=0.1, max_iters=1000)
sag_model.fit(X, y)

# Предсказание
X_new = np.array([[0], [2]])
predictions = sag_model.predict(X_new)

print("Предсказания:", predictions)
print("Веса:", sag_model.weights)
print("Смещение:", sag_model.bias)
