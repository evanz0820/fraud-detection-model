import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartData, ChartType } from 'chart.js';
import { Chart, registerables } from 'chart.js';
import { Subscription } from 'rxjs';
import { 
  FraudDetectionService, 
  DashboardStats, 
  ModelMetrics, 
  ModelInfo,
  TrainingHistory,
  SimulatedTransaction,
  TransactionActivity
} from './services/fraud-detection.service';

Chart.register(...registerables);

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  providers: [FraudDetectionService],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Fraud Detection Dashboard';
  
  // Data
  dashboardStats: DashboardStats | null = null;
  modelMetrics: ModelMetrics | null = null;
  modelInfo: ModelInfo | null = null;
  trainingHistory: TrainingHistory | null = null;
  recentTransactions: SimulatedTransaction[] = [];
  
  // UI State
  isLoading = true;
  isStreaming = false;
  apiStatus: 'connected' | 'disconnected' | 'checking' = 'checking';
  
  // Subscriptions
  private transactionSub?: Subscription;
  
  // Chart configurations
  // Confusion Matrix Chart
  confusionMatrixData: ChartData<'bar'> = {
    labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
    datasets: [{
      label: 'Count',
      data: [0, 0, 0, 0],
      backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6']
    }]
  };

  confusionMatrixOptions: ChartConfiguration<'bar'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Confusion Matrix', color: '#fff' }
    },
    scales: {
      y: { 
        beginAtZero: true,
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: {
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      }
    }
  };

  // Training Loss Chart
  trainingLossData: ChartData<'line'> = {
    labels: [],
    datasets: [{
      label: 'Training Loss',
      data: [],
      borderColor: '#8b5cf6',
      backgroundColor: 'rgba(139, 92, 246, 0.1)',
      fill: true,
      tension: 0.4
    }]
  };

  trainingLossOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, labels: { color: '#fff' } },
      title: { display: true, text: 'Training Loss Over Epochs', color: '#fff' }
    },
    scales: {
      y: { 
        beginAtZero: true, 
        title: { display: true, text: 'Loss', color: '#fff' },
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: { 
        title: { display: true, text: 'Epoch', color: '#fff' },
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      }
    }
  };

  // Performance Metrics Radar Chart
  performanceRadarData: ChartData<'radar'> = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    datasets: [{
      label: 'Model Performance',
      data: [0, 0, 0, 0, 0],
      backgroundColor: 'rgba(59, 130, 246, 0.2)',
      borderColor: '#3b82f6',
      pointBackgroundColor: '#3b82f6'
    }]
  };

  performanceRadarOptions: ChartConfiguration<'radar'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { 
        display: true, 
        text: 'Model Performance Metrics',
        color: '#fff'
      }
    },
    scales: {
      r: {
        min: 0,
        max: 1,
        beginAtZero: true,
        angleLines: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.15)'
        },
        pointLabels: {
          color: '#fff',
          font: {
            size: 12
          }
        },
        ticks: {
          display: false
        }
      }
    }
  };

  // Hourly Distribution Chart
  hourlyDistributionData: ChartData<'bar'> = {
    labels: Array.from({length: 24}, (_, i) => `${i}:00`),
    datasets: [
      {
        label: 'Normal Transactions',
        data: [],
        backgroundColor: '#10b981'
      },
      {
        label: 'Fraudulent',
        data: [],
        backgroundColor: '#ef4444'
      }
    ]
  };

  hourlyDistributionOptions: ChartConfiguration<'bar'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, labels: { color: '#fff' } },
      title: { display: true, text: 'Hourly Transaction Distribution', color: '#fff' }
    },
    scales: {
      x: { 
        stacked: true,
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      y: { 
        stacked: true, 
        beginAtZero: true,
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      }
    }
  };

  // Risk Distribution Doughnut
  riskDistributionData: ChartData<'doughnut'> = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [{
      data: [0, 0, 0],
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
    }]
  };

  riskDistributionOptions: ChartConfiguration<'doughnut'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: 'bottom', labels: { color: '#fff' } },
      title: { display: true, text: 'Risk Distribution', color: '#fff' }
    }
  };

  // Precision-Recall Curve
  prCurveData: ChartData<'line'> = {
    labels: [],
    datasets: [{
      label: 'Precision-Recall Curve',
      data: [],
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      fill: true,
      tension: 0.4
    }]
  };

  prCurveOptions: ChartConfiguration<'line'>['options'] = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Precision-Recall Curve', color: '#fff' }
    },
    scales: {
      y: { 
        min: 0, 
        max: 1, 
        title: { display: true, text: 'Precision', color: '#fff' },
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: { 
        min: 0, 
        max: 1, 
        title: { display: true, text: 'Recall', color: '#fff' },
        ticks: { color: 'rgba(255,255,255,0.7)' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      }
    }
  };

  constructor(private fraudService: FraudDetectionService) {}

  ngOnInit(): void {
    this.checkApiConnection();
    this.loadAllData();
  }

  ngOnDestroy(): void {
    this.stopTransactionStream();
  }

  checkApiConnection(): void {
    this.apiStatus = 'checking';
    this.fraudService.checkHealth().subscribe(response => {
      this.apiStatus = response.status === 'healthy' ? 'connected' : 'disconnected';
    });
  }

  loadAllData(): void {
    this.isLoading = true;

    // Load dashboard stats
    this.fraudService.getDashboardStats().subscribe(stats => {
      this.dashboardStats = stats;
      this.updateDashboardCharts(stats);
    });

    // Load model metrics
    this.fraudService.getModelMetrics().subscribe(metrics => {
      this.modelMetrics = metrics;
      this.updateMetricsCharts(metrics);
    });

    // Load model info
    this.fraudService.getModelInfo().subscribe(info => {
      this.modelInfo = info;
    });

    // Load training history
    this.fraudService.getTrainingHistory().subscribe(history => {
      this.trainingHistory = history;
      this.updateTrainingChart(history);
      this.isLoading = false;
    });
  }

  updateDashboardCharts(stats: DashboardStats): void {
    // Update confusion matrix
    this.confusionMatrixData = {
      ...this.confusionMatrixData,
      datasets: [{
        ...this.confusionMatrixData.datasets[0],
        data: [
          stats.confusion_matrix.true_negative,
          stats.confusion_matrix.false_positive,
          stats.confusion_matrix.false_negative,
          stats.confusion_matrix.true_positive
        ]
      }]
    };

    // Update hourly distribution
    this.hourlyDistributionData = {
      ...this.hourlyDistributionData,
      datasets: [
        {
          ...this.hourlyDistributionData.datasets[0],
          data: stats.hourly_distribution.map(h => h.transactions - h.fraud)
        },
        {
          ...this.hourlyDistributionData.datasets[1],
          data: stats.hourly_distribution.map(h => h.fraud)
        }
      ]
    };

    // Update risk distribution
    this.riskDistributionData = {
      ...this.riskDistributionData,
      datasets: [{
        ...this.riskDistributionData.datasets[0],
        data: [stats.risk_distribution.low, stats.risk_distribution.medium, stats.risk_distribution.high]
      }]
    };
  }

  updateMetricsCharts(metrics: ModelMetrics): void {
    // Update performance radar
    this.performanceRadarData = {
      ...this.performanceRadarData,
      datasets: [{
        ...this.performanceRadarData.datasets[0],
        data: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.roc_auc]
      }]
    };

    // Update PR curve
    if (metrics.precision_recall_curve) {
      const prData = metrics.precision_recall_curve.recall.map((r, i) => ({
        x: r,
        y: metrics.precision_recall_curve.precision[i]
      }));
      
      this.prCurveData = {
        labels: metrics.precision_recall_curve.recall.map(r => r.toFixed(2)),
        datasets: [{
          ...this.prCurveData.datasets[0],
          data: metrics.precision_recall_curve.precision
        }]
      };
    }
  }

  updateTrainingChart(history: TrainingHistory): void {
    this.trainingLossData = {
      labels: history.history.map(h => h.epoch.toString()),
      datasets: [{
        ...this.trainingLossData.datasets[0],
        data: history.history.map(h => h.loss)
      }]
    };
  }

  toggleTransactionStream(): void {
    if (this.isStreaming) {
      this.stopTransactionStream();
    } else {
      this.startTransactionStream();
    }
  }

  startTransactionStream(): void {
    this.isStreaming = true;
    this.transactionSub = this.fraudService.getTransactionStream(1500).subscribe(txn => {
      this.recentTransactions.unshift(txn);
      if (this.recentTransactions.length > 20) {
        this.recentTransactions.pop();
      }
    });
  }

  stopTransactionStream(): void {
    this.isStreaming = false;
    this.transactionSub?.unsubscribe();
  }

  getRiskClass(riskLevel: string): string {
    switch (riskLevel.toUpperCase()) {
      case 'HIGH': return 'risk-high';
      case 'MEDIUM': return 'risk-medium';
      default: return 'risk-low';
    }
  }

  formatPercentage(value: number): string {
    return (value * 100).toFixed(2) + '%';
  }

  formatNumber(value: number): string {
    return value.toLocaleString();
  }

  formatCurrency(value: number): string {
    return '$' + value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  formatTime(timestamp: string): string {
    return new Date(timestamp).toLocaleTimeString();
  }
}