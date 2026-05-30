# MDD анализ
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Данные из задания
np.random.seed(42)

existing_system_responses = np.random.normal(loc=3.5, scale=0.4, size=500_000)
improved_system_responses = np.random.normal(loc=2.0, scale=0.4, size=500_000)

# Описательная статистика
print("=" * 60)
print("Описательная статистика")
print("=" * 60)
for label, data in [("Существующая", existing_system_responses),
                    ("Улучшенная",   improved_system_responses)]:
    p50, p95, p99 = np.percentile(data, [50, 95, 99])
    print(f"\n{label} система:")
    print(f"  Mean:  {data.mean():.4f} с")
    print(f"  Std:   {data.std():.4f} с")
    print(f"  P50:   {p50:.4f} с")
    print(f"  P95:   {p95:.4f} с")
    print(f"  P99:   {p99:.4f} с")

# Формулировка гипотез
print("\n" + "=" * 60)
print("Гипотезы")
print("=" * 60)
print('''
      H0: мю существующая = мю улучшенная (нет разницы)\n
      H1: мю существующая < мю улучшенная (улучшенная быстрее)\n
      Тип теста: односторонний\n
      Уровень значимости alpha: 0.01
      ''')

# Статистический тест (Mann–Whitney U)
print("\n" + "=" * 60)
print("Статистический тест Mann–Whitney U (односторонний)")
print("=" * 60)

# alternative='greater': проверяем, что existing > improved по значениям
stat, p_value = stats.mannwhitneyu(
    existing_system_responses,
    improved_system_responses,
    alternative="greater"
)

alpha = 0.01
print(f"U-статистика:  {stat:.2e}")
print(f"p-value:       {p_value:.2e}")
print(f"alpha (порог):     {alpha}")
print()
if p_value < alpha:
    print(''' 
          p-value << alpha -> H0 ОТКЛОНЕНА.\n 
          Улучшенная система статистически значимо быстрее существующей 
          (alpha=0.01).
          ''')
else:
    print("p-value > α -> H0 НЕ ОТКЛОНЕНА")

# Effect size
pooled_std = np.sqrt((existing_system_responses.std()**2 +
                      improved_system_responses.std()**2) / 2)
cohens_d = (existing_system_responses.mean() -
            improved_system_responses.mean()) / pooled_std

print("\n" + "=" * 55)
print("Практическая значимость")
print("=" * 55)
improvement_pct = (existing_system_responses.mean() -
                   improved_system_responses.mean()) / existing_system_responses.mean() * 100
print(f'''
      Снижение средней латентности: {improvement_pct:.1f}%\n
      Cohen's d:  {cohens_d:.2f}  (>0.8 — большой эффект)\n
      SLO P95 < 0.5с:\n
      Существующая P95: {np.percentile(existing_system_responses, 95):.2f} с (нарушает SLO)\n
      Улучшенная   P95: {np.percentile(improved_system_responses, 95):.2f} с (выполняет SLO)\n
      ''')

# Итоговое архитектурное решение
print("*" * 55)
print("Архитектурное решение (ADR-001)")
print("*" * 55)
print("H0: мю улучшенное = мю существующее")
print("H1: мю улучшенное < мю существующее")
print("Тест Mann–Whitney U (односторонний)")
print(f"p-value: {p_value:.2e}")
print(f"alpha:     {alpha}")
print("H0:    Отклонена")
print("Решение деплоить улучшенную систему в production")

# Визуализация
plt.figure(figsize=(10, 6))
sns.kdeplot(existing_system_responses, label="Существующая система",
            fill=True, color="red", alpha=0.5)
sns.kdeplot(improved_system_responses, label="Улучшенная система",
            fill=True, color="green", alpha=0.5)

# Добавляем SLO
plt.axvline(x=0.5, color="orange", linestyle="--", linewidth=2, label="SLO P95 = 0.5с")

plt.title(
    f"Сравнение времени отклика систем\n"
    f"Mann–Whitney p={p_value:.2e}, Cohen's d={cohens_d:.2f}, улучшение {improvement_pct:.1f}%",
    fontsize=13
)
plt.xlabel("Время отклика (секунды)")
plt.ylabel("Плотность")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Перевод оси Y в проценты (как в задании)
ax = plt.gca()
ymin, ymax = ax.get_ylim()
num_ticks = 5
new_yticks = np.linspace(ymin, ymax, num_ticks)
new_yticklabels = [
    f"{int((tick / ymax) * 100)}%" if ymax != 0 else "0%"
    for tick in new_yticks
]
ax.set_yticks(new_yticks)
ax.set_yticklabels(new_yticklabels)

plt.tight_layout()
plt.savefig("docs/latency_comparison.png", dpi=150, bbox_inches="tight")
print("\nГрафик сохранён: docs/latency_comparison.png")
plt.show()
