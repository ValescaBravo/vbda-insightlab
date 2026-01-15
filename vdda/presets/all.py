"""
===============================================================================
INSIGHTLAB — MASTER DOMAIN PRESETS (8 INDUSTRY PRESETS)
===============================================================================

Este archivo contiene TODOS los presets de dominio de InsightLab:

    1. Marketing Analytics
    2. Ecommerce Forecasting
    3. Customer Churn
    4. Product Analytics
    5. A/B Testing
    6. Social Sciences
    7. Healthcare Analytics
    8. Real Estate Analytics

Cada preset incluye:
    • phrases         → lenguaje narrativo especializado
    • kpis            → indicadores del dominio
    • benchmarks      → estándares orientativos
    • interpretation  → plantillas interpretativas
    • defaults        → parámetros base para análisis

Comentarios en español.
Narrativa en inglés.
===============================================================================
"""

PRESETS = {

# =============================================================================
# 1. MARKETING ANALYTICS
# =============================================================================
"marketing": {

    "phrases": {
        "intro": (
            "This insight is framed within digital marketing performance. "
            "It highlights audience behaviour, bidding efficiency and ROI."
        ),
        "risk": (
            "Platform-level volatility, algorithmic shifts, or seasonal events "
            "may influence these results."
        ),
        "footer": (
            "Interpret results considering creative fatigue, audience overlap, "
            "frequency capping and campaign pacing."
        ),
    },

    "kpis": {
        "ctr": "Click-Through Rate",
        "cvr": "Conversion Rate",
        "cpc": "Cost per Click",
        "cpa": "Cost per Acquisition",
        "impressions": "Impressions",
        "frequency": "Ad Frequency",
        "roas": "Return on Ad Spend",
    },

    "benchmarks": {
        "ctr_good": 0.015,
        "ctr_excellent": 0.03,
        "cvr_good": 0.02,
        "roas_good": 3.0,
        "frequency_ceiling": 7,
    },

    "interpretation": {
        "high_ctr": "Creative resonance and audience alignment appear strong.",
        "low_ctr": "Messaging may be misaligned or creative fatigue may be present.",
        "high_cvr": "Landing page, intent and offer appear well-aligned.",
        "low_cvr": "Conversion friction or expectation mismatch detected.",
        "high_frequency": "User saturation or creative wear-out likely.",
        "low_roas": "Spend inefficiency or funnel misalignment suspected."
    },

    "defaults": {
        "correlation_threshold": 0.5,
        "significance_level": 0.05
    }
},

# =============================================================================
# 2. ECOMMERCE FORECASTING
# =============================================================================
"ecommerce": {

    "phrases": {
        "intro": (
            "Analysis focuses on ecommerce demand behaviour and revenue trends."
        ),
        "risk": (
            "Seasonality, stock availability and promotional events may distort patterns."
        ),
        "footer": (
            "Interpret sales with consideration of competitor pricing, user intent "
            "and macroeconomic trends."
        ),
    },

    "kpis": {
        "aov": "Average Order Value",
        "cr": "Conversion Rate",
        "revenue": "Revenue",
        "units": "Units Sold",
        "repeat_rate": "Repeat Purchase Rate",
        "traffic": "Site Traffic"
    },

    "benchmarks": {
        "aov_solid": 50,
        "repeat_good": 0.25,
        "conversion_expected": 0.02
    },

    "interpretation": {
        "high_aov": "Users are purchasing more per transaction.",
        "low_conversion": "Funnel friction or pricing mismatch likely.",
        "traffic_spike": "Campaign or referral lift detected."
    },

    "defaults": {
        "forecast_horizon": 30,
        "significance_level": 0.05
    }
},

# =============================================================================
# 3. CUSTOMER CHURN
# =============================================================================
"churn": {

    "phrases": {
        "intro": (
            "This insight reflects customer retention and churn dynamics."
        ),
        "risk": (
            "External shocks, competition or missing behavioural data may distort results."
        ),
        "footer": (
            "Churn must be interpreted alongside cohort structure and engagement patterns."
        )
    },

    "kpis": {
        "churn_rate": "Churn Rate",
        "retention": "Retention Rate",
        "ltv": "Lifetime Value",
        "engagement": "Engagement Score"
    },

    "benchmarks": {
        "good_retention": 0.85,
        "acceptable_churn": 0.15
    },

    "interpretation": {
        "high_engagement": "Customers show healthy ongoing usage.",
        "churn_spike": "Value perception or satisfaction drop suspected.",
        "low_ltv": "Revenue per customer below sustainable levels."
    },

    "defaults": {
        "window": 30,
        "significance_level": 0.05
    }
},

# =============================================================================
# 4. PRODUCT ANALYTICS
# =============================================================================
"product": {

    "phrases": {
        "intro": (
            "Insight is contextualized within product usage and behavioural analytics."
        ),
        "risk": (
            "Instrumented events may not fully reflect the underlying user intent."
        ),
        "footer": (
            "Consider funnel friction, value delivery, and retention cohorts."
        )
    },

    "kpis": {
        "wau": "Weekly Active Users",
        "mau": "Monthly Active Users",
        "stickiness": "DAU/MAU Ratio",
        "session_length": "Session Length",
        "feature_adoption": "Feature Adoption Rate"
    },

    "benchmarks": {
        "stickiness_good": 0.2,
        "adoption_good": 0.3
    },

    "interpretation": {
        "high_stickiness": "Strong habitual product use detected.",
        "low_adoption": "Feature may lack clarity or value delivery.",
        "high_session_length": "Deep product engagement likely."
    },

    "defaults": {
        "correlation_threshold": 0.4
    }
},

# =============================================================================
# 5. A/B TESTING
# =============================================================================
"ab_testing": {

    "phrases": {
        "intro": (
            "This insight is based on controlled experimentation (A/B testing)."
        ),
        "risk": (
            "Sample imbalance, novelty effects, or insufficient run duration "
            "may bias results."
        ),
        "footer": (
            "Confirm test validity: randomization quality, sample size, and time window."
        )
    },

    "kpis": {
        "uplift": "Uplift",
        "cr": "Conversion Rate",
        "delta": "Difference in Means",
        "impact": "Absolute Impact"
    },

    "benchmarks": {},

    "interpretation": {
        "significant": "Statistically meaningful uplift detected.",
        "non_significant": "No evidence of meaningful difference.",
        "effect_large": "Practical significance is high."
    },

    "defaults": {
        "significance_level": 0.05
    }
},

# =============================================================================
# 6. SOCIAL SCIENCES
# =============================================================================
"social_sciences": {

    "phrases": {
        "intro": (
            "This analysis is framed within behavioural and social phenomena."
        ),
        "risk": (
            "Sampling bias or self-reporting may impact validity."
        ),
        "footer": (
            "Interpret findings considering cultural, demographic and contextual factors."
        )
    },

    "kpis": {
        "mean_diff": "Difference in Means",
        "correlation": "Correlation Coefficient",
        "effect_size": "Effect Size"
    },

    "benchmarks": {},

    "interpretation": {
        "medium_effect": "Meaningful behavioural difference detected.",
        "weak_correlation": "Weak association; caution advised.",
        "strong_correlation": "Potentially meaningful relationship."
    },

    "defaults": {
        "significance_level": 0.05
    }
},

# =============================================================================
# 7. HEALTHCARE ANALYTICS
# =============================================================================
"healthcare": {

    "phrases": {
        "intro": (
            "Insight contextualized within healthcare data, risk, and outcome modelling."
        ),
        "risk": (
            "Ethical considerations, sampling bias, and patient variability "
            "must be carefully considered."
        ),
        "footer": (
            "Interpret findings alongside clinical relevance and patient safety."
        )
    },

    "kpis": {
        "readmission": "Readmission Rate",
        "mortality": "Mortality Rate",
        "risk_score": "Risk Score",
        "treatment_effect": "Estimated Treatment Effect"
    },

    "benchmarks": {},

    "interpretation": {
        "high_risk": "Patient group shows elevated risk indicators.",
        "treatment_effect_positive": "Intervention may be beneficial.",
        "functional_decline": "Potential deterioration detected."
    },

    "defaults": {
        "significance_level": 0.05
    }
},

# =============================================================================
# 8. REAL ESTATE ANALYTICS
# =============================================================================
"real_estate": {

    "phrases": {
        "intro": (
            "Insight framed within property value behaviour and demand dynamics."
        ),
        "risk": (
            "Location-specific trends, macroeconomics, and policy changes may affect results."
        ),
        "footer": (
            "Interpret real estate metrics considering supply, demand and economic cycles."
        )
    },

    "kpis": {
        "price": "Property Price",
        "sqft": "Square Footage",
        "dom": "Days on Market",
        "inventory": "Inventory Level"
    },

    "benchmarks": {},

    "interpretation": {
        "price_spike": "Demand surge in the segment.",
        "high_dom": "Potential oversupply or pricing issue.",
        "inventory_low": "Seller’s market conditions."
    },

    "defaults": {
        "significance_level": 0.05
    }
}

}  # END PRESETS
