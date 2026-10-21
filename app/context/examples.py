"""Fictional historical estimates used as static few-shot context."""

HISTORICAL_ESTIMATES = (
    {
        'name': 'Portal de reservas',
        'hourly_rate_eur': 50,
        'tasks': (
            {'name': 'Análisis', 'hours': 8, 'cost_eur': 400},
            {'name': 'API y base de datos', 'hours': 24, 'cost_eur': 1200},
            {'name': 'Interfaz', 'hours': 20, 'cost_eur': 1000},
            {'name': 'Pruebas y despliegue', 'hours': 12, 'cost_eur': 600},
        ),
        'total_hours': 64,
        'total_cost_eur': 3200,
        'assumption': 'autenticación ya disponible',
    },
    {
        'name': 'Panel interno de informes',
        'hourly_rate_eur': 50,
        'tasks': (
            {'name': 'Análisis', 'hours': 6, 'cost_eur': 300},
            {'name': 'Integración de datos', 'hours': 16, 'cost_eur': 800},
            {'name': 'Visualización', 'hours': 18, 'cost_eur': 900},
            {'name': 'Pruebas y despliegue', 'hours': 10, 'cost_eur': 500},
        ),
        'total_hours': 50,
        'total_cost_eur': 2500,
        'assumption': 'una única fuente de datos estable',
    },
)


def render_examples() -> str:
    """Format fictional tasks, totals and assumptions as Spanish few-shot context.

    These examples guide estimation rather than establish universal rates.
    """
    sections = []
    for number, example in enumerate(HISTORICAL_ESTIMATES, start=1):
        lines = [f"Ejemplo {number} — {example['name']}, tarifa {example['hourly_rate_eur']} €/hora:"]
        lines.extend(
            f"- {task['name']}: {task['hours']} horas, {task['cost_eur']} €."
            for task in example['tasks']
        )
        lines.append(
            f"Total: {example['total_hours']} horas, {example['total_cost_eur']} €. "
            f"Supuesto: {example['assumption']}."
        )
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections)
