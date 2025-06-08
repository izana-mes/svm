import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class SVMBWOOptimizer:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.history = []
    
    def load_data(self, file_path, target_col, sep=','):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(file_path, sep=sep)
            self.X = self.data.drop(target_col, axis=1).values
            self.y = self.data[target_col].values
            print("Data loaded successfully!")
            print(f"Shape of features: {self.X.shape}")
            print(f"Shape of target: {self.y.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess and split data"""
        try:
            # Scale features
            self.X = self.scaler.fit_transform(self.X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)
            
            print("Data preprocessing completed!")
            print(f"Train set: {self.X_train.shape}, {self.y_train.shape}")
            print(f"Test set: {self.X_test.shape}, {self.y_test.shape}")
            return True
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return False
    
    def evaluate_svm(self, C, gamma):
        """Evaluate SVM with given parameters"""
        svm = SVC(C=C, gamma=gamma)
        scores = cross_val_score(svm, self.X_train, self.y_train, cv=5, scoring='accuracy')
        return np.mean(scores)
    
    def run_bwo(self, pop_size=10, max_iter=50, pc=0.6, pm=0.4, C_bounds=(0.1, 100), gamma_bounds=(0.0001, 10)):
        """Run Black Widow Optimization algorithm"""
        # Initialize population
        population = []
        for _ in range(pop_size):
            C = np.random.uniform(*C_bounds)
            gamma = np.random.uniform(*gamma_bounds)
            fitness = self.evaluate_svm(C, gamma)
            population.append({'C': C, 'gamma': gamma, 'fitness': fitness})
        
        best_solution = max(population, key=lambda x: x['fitness'])
        self.history.append(best_solution['fitness'])
        
        for iteration in range(max_iter):
            # Sort population by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Mating phase
            new_population = []
            for i in range(0, pop_size-1, 2):
                if np.random.rand() < pc:
                    # Arithmetic crossover
                    alpha = np.random.rand()
                    child1 = {
                        'C': alpha * population[i]['C'] + (1-alpha) * population[i+1]['C'],
                        'gamma': alpha * population[i]['gamma'] + (1-alpha) * population[i+1]['gamma']
                    }
                    child2 = {
                        'C': alpha * population[i+1]['C'] + (1-alpha) * population[i]['C'],
                        'gamma': alpha * population[i+1]['gamma'] + (1-alpha) * population[i]['gamma']
                    }
                    
                    child1['fitness'] = self.evaluate_svm(child1['C'], child1['gamma'])
                    child2['fitness'] = self.evaluate_svm(child2['C'], child2['gamma'])
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([population[i], population[i+1]])
            
            # Cannibalism - keep only the best individuals
            new_population.sort(key=lambda x: x['fitness'], reverse=True)
            population = new_population[:pop_size]
            
            # Mutation
            for i in range(pop_size):
                if np.random.rand() < pm:
                    mutated = population[i].copy()
                    if np.random.rand() < 0.5:
                        mutated['C'] = np.clip(mutated['C'] * np.random.normal(1, 0.1), *C_bounds)
                    else:
                        mutated['gamma'] = np.clip(mutated['gamma'] * np.random.normal(1, 0.1), *gamma_bounds)
                    mutated['fitness'] = self.evaluate_svm(mutated['C'], mutated['gamma'])
                    population[i] = mutated
            
            # Update best solution
            current_best = max(population, key=lambda x: x['fitness'])
            if current_best['fitness'] > best_solution['fitness']:
                best_solution = current_best.copy()
            
            self.history.append(best_solution['fitness'])
            print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {best_solution['fitness']:.4f}")
        
        self.best_params = best_solution
        return best_solution
    
    def evaluate_on_test(self):
        """Evaluate the best model on test set"""
        if self.best_params is None:
            print("Please run optimization first!")
            return None
        
        best_svm = SVC(C=self.best_params['C'], gamma=self.best_params['gamma'])
        best_svm.fit(self.X_train, self.y_train)
        y_pred = best_svm.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print("\nFinal Evaluation:")
        print(f"Best C: {self.best_params['C']:.4f}")
        print(f"Best gamma: {self.best_params['gamma']:.4f}")
        print(f"Cross-validation Accuracy: {self.best_params['fitness']:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def plot_convergence(self):
        """Plot the convergence of fitness values"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, 'b-o', linewidth=2, markersize=8)
        plt.title('BWO Optimization Convergence', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Best Fitness (Accuracy)', fontsize=14)
        plt.grid(True)
        plt.show()

def main():
    print("SVM Optimization using Black Widow Algorithm")
    print("------------------------------------------")
    
    optimizer = SVMBWOOptimizer()
    
    # Step 1: Load data
    file_path = input("Enter path to your CSV file: ")
    target_col = input("Enter the name of target column: ")
    
    if not optimizer.load_data(file_path, target_col):
        return
    
    # Step 2: Preprocess data
    if not optimizer.preprocess_data():
        return
    
    # Step 3: Run optimization
    print("\nRunning Black Widow Optimization...")
    pop_size = int(input("Enter population size (default 10): ") or 10)
    max_iter = int(input("Enter maximum iterations (default 50): ") or 50)
    
    optimizer.run_bwo(pop_size=pop_size, max_iter=max_iter)
    
    # Step 4: Evaluate results
    optimizer.evaluate_on_test()
    optimizer.plot_convergence()
    
    # Save best parameters
    save_results = input("Do you want to save the best parameters? (y/n): ").lower()
    if save_results == 'y':
        with open('best_params.txt', 'w') as f:
            f.write(f"C: {optimizer.best_params['C']}\n")
            f.write(f"gamma: {optimizer.best_params['gamma']}\n")
            f.write(f"Validation Accuracy: {optimizer.best_params['fitness']}\n")
            f.write(f"Test Accuracy: {optimizer.evaluate_on_test()}\n")
        print("Results saved to best_params.txt")

if __name__ == "__main__":
    main()