import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, of } from 'rxjs';
import { map, catchError, switchMap } from 'rxjs/operators';

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  average_precision: number;
  optimal_threshold: number;
  confusion_matrix: number[][];
  precision_recall_curve: {
    precision: number[];
    recall: number[];
  };
}

export interface DashboardStats {
  overview: {
    total_transactions: number;
    total_fraud_detected: number;
    fraud_rate: number;
    amount_saved: number;
    model_accuracy: number;
  };
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
  };
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  recent_activity: TransactionActivity[];
  hourly_distribution: HourlyData[];
  risk_distribution: {
    low: number;
    medium: number;
    high: number;
  };
}

export interface TransactionActivity {
  id: string;
  timestamp: string;
  amount: number;
  is_fraud: boolean;
  risk_level: string;
  confidence: number;
}

export interface HourlyData {
  hour: number;
  transactions: number;
  fraud: number;
}

export interface PredictionResult {
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  confidence: number;
  amount: number;
}

export interface SimulatedTransaction {
  transaction: {
    id: string;
    timestamp: string;
    amount: number;
    features?: Record<string, number>;
  };
  prediction: {
    is_fraud: boolean;
    fraud_probability: number;
    risk_level: string;
    processing_time_ms: number;
  };
}

export interface ModelInfo {
  model_type: string;
  trained_at: string;
  dataset_info: {
    total_transactions: number;
    fraud_count: number;
    normal_count: number;
    fraud_percentage: number;
    features: string[];
    time_range: number[];
    amount_range: number[];
  };
  architecture: {
    type: string;
    layers: number[];
    activation: string;
    dropout: number;
  };
}

export interface TrainingHistory {
  history: {
    epoch: number;
    loss: number;
    lr: number;
  }[];
}

@Injectable({
  providedIn: 'root'
})
export class FraudDetectionService {
  private apiUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  // Health check
  checkHealth(): Observable<any> {
    return this.http.get(`${this.apiUrl}/health`).pipe(
      catchError(error => of({ status: 'error', error: error.message }))
    );
  }

  // Get model information
  getModelInfo(): Observable<ModelInfo> {
    return this.http.get<ModelInfo>(`${this.apiUrl}/model/info`).pipe(
      catchError(error => {
        console.error('Error fetching model info:', error);
        return of(this.getMockModelInfo());
      })
    );
  }

  // Get model metrics
  getModelMetrics(): Observable<ModelMetrics> {
    return this.http.get<ModelMetrics>(`${this.apiUrl}/model/metrics`).pipe(
      catchError(error => {
        console.error('Error fetching metrics:', error);
        return of(this.getMockMetrics());
      })
    );
  }

  // Get training history
  getTrainingHistory(): Observable<TrainingHistory> {
    return this.http.get<TrainingHistory>(`${this.apiUrl}/model/training-history`).pipe(
      catchError(error => {
        console.error('Error fetching training history:', error);
        return of(this.getMockTrainingHistory());
      })
    );
  }

  // Get dashboard statistics
  getDashboardStats(): Observable<DashboardStats> {
    return this.http.get<DashboardStats>(`${this.apiUrl}/dashboard/stats`).pipe(
      catchError(error => {
        console.error('Error fetching dashboard stats:', error);
        return of(this.getMockDashboardStats());
      })
    );
  }

  // Make prediction
  predict(transaction: any): Observable<PredictionResult> {
    return this.http.post<PredictionResult>(`${this.apiUrl}/predict`, transaction).pipe(
      catchError(error => {
        console.error('Error making prediction:', error);
        return of(this.getMockPrediction());
      })
    );
  }

  // Simulate transaction (for live demo)
  simulateTransaction(): Observable<SimulatedTransaction> {
    return this.http.get<SimulatedTransaction>(`${this.apiUrl}/simulate/transaction`).pipe(
      catchError(error => {
        console.error('Error simulating transaction:', error);
        return of(this.getMockSimulatedTransaction());
      })
    );
  }

  // Get real-time transaction stream
  getTransactionStream(intervalMs: number = 2000): Observable<SimulatedTransaction> {
    return interval(intervalMs).pipe(
      switchMap(() => this.simulateTransaction())
    );
  }

  // Mock data methods for demo mode
  private getMockModelInfo(): ModelInfo {
    return {
      model_type: 'nn',
      trained_at: new Date().toISOString(),
      dataset_info: {
        total_transactions: 284807,
        fraud_count: 492,
        normal_count: 284315,
        fraud_percentage: 0.172,
        features: ['Time', 'V1', 'V2', '...', 'V28', 'Amount'],
        time_range: [0, 172792],
        amount_range: [0, 25691.16]
      },
      architecture: {
        type: 'Deep Neural Network',
        layers: [128, 64, 32],
        activation: 'ReLU',
        dropout: 0.3
      }
    };
  }

  private getMockMetrics(): ModelMetrics {
    return {
      accuracy: 0.9994,
      precision: 0.9412,
      recall: 0.8163,
      f1_score: 0.8744,
      roc_auc: 0.9876,
      average_precision: 0.8234,
      optimal_threshold: 0.42,
      confusion_matrix: [[56855, 9], [18, 80]],
      precision_recall_curve: {
        precision: Array.from({length: 100}, (_, i) => 1 - i * 0.005),
        recall: Array.from({length: 100}, (_, i) => i * 0.01)
      }
    };
  }

  private getMockTrainingHistory(): TrainingHistory {
    return {
      history: Array.from({length: 50}, (_, i) => ({
        epoch: i + 1,
        loss: 0.5 * Math.exp(-i * 0.05) + 0.01,
        lr: 0.001 * Math.pow(0.95, Math.floor(i / 10))
      }))
    };
  }

  private getMockDashboardStats(): DashboardStats {
    return {
      overview: {
        total_transactions: 284807,
        total_fraud_detected: 492,
        fraud_rate: 0.172,
        amount_saved: 247832.45,
        model_accuracy: 99.94
      },
      model_performance: {
        accuracy: 0.9994,
        precision: 0.9412,
        recall: 0.8163,
        f1_score: 0.8744,
        roc_auc: 0.9876
      },
      confusion_matrix: {
        true_negative: 56855,
        false_positive: 9,
        false_negative: 18,
        true_positive: 80
      },
      recent_activity: this.generateMockActivity(),
      hourly_distribution: this.generateMockHourlyData(),
      risk_distribution: {
        low: 92.5,
        medium: 6.3,
        high: 1.2
      }
    };
  }

  private generateMockActivity(): TransactionActivity[] {
    const activities: TransactionActivity[] = [];
    const now = new Date();
    
    for (let i = 0; i < 10; i++) {
      const isFraud = Math.random() < 0.15;
      activities.push({
        id: `TXN${Math.floor(100000 + Math.random() * 900000)}`,
        timestamp: new Date(now.getTime() - Math.random() * 3600000).toISOString(),
        amount: Math.round(Math.random() * 2000 * 100) / 100,
        is_fraud: isFraud,
        risk_level: isFraud ? 'HIGH' : Math.random() > 0.8 ? 'MEDIUM' : 'LOW',
        confidence: Math.round((0.7 + Math.random() * 0.29) * 100) / 100
      });
    }
    
    return activities.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  private generateMockHourlyData(): HourlyData[] {
    return Array.from({length: 24}, (_, i) => ({
      hour: i,
      transactions: Math.floor(100 + Math.random() * 400),
      fraud: Math.floor(Math.random() * 5)
    }));
  }

  private getMockPrediction(): PredictionResult {
    const isFraud = Math.random() < 0.1;
    return {
      is_fraud: isFraud,
      fraud_probability: isFraud ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3,
      risk_level: isFraud ? 'HIGH' : 'LOW',
      confidence: 0.85 + Math.random() * 0.14,
      amount: Math.round(Math.random() * 1000 * 100) / 100
    };
  }

  private getMockSimulatedTransaction(): SimulatedTransaction {
    const isFraud = Math.random() < 0.1;
    return {
      transaction: {
        id: `TXN${Math.floor(100000 + Math.random() * 900000)}`,
        timestamp: new Date().toISOString(),
        amount: Math.round(Math.random() * 1000 * 100) / 100
      },
      prediction: {
        is_fraud: isFraud,
        fraud_probability: isFraud ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3,
        risk_level: isFraud ? 'HIGH' : Math.random() > 0.9 ? 'MEDIUM' : 'LOW',
        processing_time_ms: Math.floor(5 + Math.random() * 45)
      }
    };
  }
}
