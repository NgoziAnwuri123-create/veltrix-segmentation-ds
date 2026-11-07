"""Feature dictionary for ds.customer_features (feature version fv1).

Single source of truth for the feature documentation, the leakage check and preprocessing.
kind:
    cat    categorical, kept as a category (one-hot for k-means, matching for k-prototypes)
    flag   boolean, used as 0/1
    count  non-negative count; log1p-transformed
    share  proportion 0-1 of the customer's activity; NULL when the customer has none of that activity
    num    other numeric; log1p-transformed when log=True
    price  stated or viewed price; combined into intent-relative price levels by preprocessing
model: True if the column is a model input. Excluded columns carry the reason in `note`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    kind: str
    model: bool
    note: str
    log: bool = False


def _shares(group: str, items: dict[str, str]) -> list[Feature]:
    return [Feature(n, group, "share", True, d) for n, d in items.items()]


FEATURES: list[Feature] = [
    # profile -------------------------------------------------------------------------------------
    Feature("customer_type", "profile", "cat", True, "Individual, Limited company (SPV) or Corporate"),
    Feature("primary_intent", "profile", "cat", True, "Buy or Rent"),
    Feature("finance_type", "profile", "cat", True, "Mortgage, cash, BTL mortgage, LISA, shared ownership, ..."),
    Feature("buyer_position", "profile", "cat", True, "First-time buyer, home mover, downsizer, landlord, ..."),
    Feature("age_band", "profile", "cat", True, "Age band; missing -> Unknown (27% missing)"),
    Feature("region_of_residence", "profile", "cat", True, "UK region of residence, or Overseas"),
    Feature("is_overseas", "profile", "flag", True, "Lives outside the UK"),
    Feature("non_gbp_currency", "profile", "flag", True, "Prefers a currency other than GBP"),
    Feature("budget_missing", "profile", "flag", True, "Did not state the budget that applies to their intent"),
    Feature("budget_gbp", "profile", "price", True, "Stated sale budget (typos fixed) -> price_level"),
    Feature("rent_budget_pcm_gbp", "profile", "price", True, "Stated rent budget per month -> price_level"),
    Feature("occupation_sector", "profile", "cat", False, "Excluded: 27% missing and not tied to a housing need"),
    Feature("gender", "profile", "cat", False, "Excluded: sensitive attribute with no use for serving customers"),
    Feature("country_iso2", "profile", "cat", False, "Excluded: 12 sparse levels; is_overseas and IP share carry it"),
    Feature("consent_marketing", "profile", "flag", False, "Excluded: a permission, not a behaviour; used for audiences"),
    Feature("tenure_days", "profile", "num", False, "Excluded: reflects when the customer joined, not who they are"),
    Feature("probable_duplicate", "quality", "flag", False, "Excluded: data-quality flag; such rows are left out of training"),
    Feature("has_leads", "presence", "flag", False, "Excluded: same information as n_leads = 0"),
    Feature("has_web", "presence", "flag", False, "Excluded: same information as n_sessions = 0"),
    Feature("has_views", "presence", "flag", False, "Excluded: same information as n_views = 0"),
    # enquiry -------------------------------------------------------------------------------------
    Feature("n_leads", "enquiry", "count", True, "Enquiries made"),
    Feature("n_enquiry_months", "enquiry", "count", True, "Distinct months with an enquiry (repeat enquirers)"),
    *_shares("enquiry", {
        "share_rent_leads": "Share of enquiries about renting",
        "ch_portal": "Share of enquiries from property portals",
        "ch_website": "Share from the Veltrix website",
        "ch_paid_search": "Share from paid search",
        "ch_social": "Share from Facebook, Instagram or TikTok",
        "ch_wechat": "Share from WeChat",
        "ch_email": "Share from the email newsletter",
        "ch_referral": "Share from referrals",
        "ch_overseas_partner": "Share from overseas agent partners",
        "ch_relocation_partner": "Share from relocation partners",
        "ch_branch": "Share from branch walk-ins",
    }),
    # contact -------------------------------------------------------------------------------------
    Feature("n_interactions", "contact", "count", False,
            "Excluded (leakage): grows as a lead progresses to offer; Spearman 0.57 with converted"),
    *_shares("contact", {
        "share_inbound": "Share of interactions started by the customer",
        "ct_phone": "Share by phone", "ct_email": "Share by email", "ct_whatsapp": "Share by WhatsApp",
        "ct_live_chat": "Share by live chat", "ct_video_call": "Share by video call",
        "ct_branch_visit": "Share by branch visit",
        "tp_mortgage": "Share about mortgages and finance", "tp_stamp_duty": "Share about stamp duty",
        "tp_rental_yield": "Share about rental yield", "tp_chain": "Share about chains and timescales",
        "tp_tenancy_terms": "Share about tenancy terms", "tp_shared_ownership": "Share about shared ownership",
        "tp_offplan_schedule": "Share about off-plan payment schedules",
        "tp_school_catchment": "Share about school catchment", "tp_retirement": "Share about retirement living",
        "tp_visa_relocation": "Share about visas and relocation", "tp_currency": "Share about currency and transfers",
    }),
    # viewings ------------------------------------------------------------------------------------
    Feature("n_site_visits", "viewings", "count", False,
            "Excluded (leakage): more viewings is part of progressing to a deal; Spearman 0.53 with converted"),
    *_shares("viewings", {
        "sv_virtual": "Share of viewings done virtually",
        "sv_proxy": "Share attended by family or solicitor on their behalf",
        "sv_open_house": "Share at open houses",
    }),
    # web -----------------------------------------------------------------------------------------
    Feature("n_sessions", "web", "count", True, "Logged-in web sessions"),
    Feature("n_active_months", "web", "count", True, "Distinct months with a session"),
    *_shares("web", {
        "dv_mobile": "Share of sessions on mobile", "dv_tablet": "Share on tablet",
        "share_night_sessions": "Share of sessions starting 00:00-06:00 UK time",
        "share_overseas_ip": "Share of sessions from a non-UK IP",
        "tr_portal": "Share arriving from a property portal", "tr_organic_search": "Share from organic search",
        "tr_paid_search": "Share from paid search", "tr_social": "Share from paid or organic social",
        "tr_direct": "Share arriving directly", "tr_email": "Share from email",
        "tr_partner": "Share from referral partners",
    }),
    Feature("avg_pages_per_session", "web", "num", True, "Average pages per session"),
    Feature("med_session_sec", "web", "num", True, "Median session length in seconds", log=True),
    # property views ------------------------------------------------------------------------------
    Feature("n_views", "views", "count", True, "Property view events"),
    Feature("n_properties_viewed", "views", "count", True, "Distinct properties viewed"),
    Feature("views_per_session", "views", "num", True, "Property events per session", log=True),
    Feature("enquiries_per_100_views", "views", "num", True, "Enquiries per 100 views (low = window shopping)", log=True),
    *_shares("views", {
        "ev_save": "Share of view events that are saves", "ev_virtual_tour": "Share that are virtual tours",
        "ev_share": "Share that are shares", "ev_floorplan": "Share that are floorplan downloads",
        "share_let_views": "Share of views on rental listings",
        "pt_flat": "Share of views on flats", "pt_maisonette": "On maisonettes", "pt_terraced": "On terraced houses",
        "pt_semi": "On semi-detached houses", "pt_detached": "On detached houses", "pt_bungalow": "On bungalows",
        "share_new_build": "Share of views on new-build homes",
        "share_offplan": "Share on off-plan or under-construction homes",
        "rg_london": "Share of views in London", "rg_south_east": "In the South East and East of England",
        "rg_north": "In the North West, North East and Yorkshire", "rg_midlands": "In the West and East Midlands",
        "rg_south_west_wales": "In the South West and Wales", "rg_scotland": "In Scotland",
    }),
    Feature("med_asking_viewed_gbp", "views", "price", True, "Median asking price viewed -> viewed_price_level"),
    Feature("med_rent_viewed_pcm_gbp", "views", "price", True, "Median rent viewed -> viewed_price_level"),
    Feature("avg_yield_viewed_pct", "views", "num", True, "Average gross yield of the areas viewed"),
    # campaigns -----------------------------------------------------------------------------------
    Feature("n_campaign_responses", "campaigns", "count", True, "Responses to marketing campaigns"),
    *_shares("campaigns", {
        "rs_form_submit": "Share of responses that are form submissions",
        "rs_negative": "Share that hide the ad or unsubscribe",
        "rc_tiktok": "Share of responses to TikTok campaigns", "rc_meta": "To Meta campaigns",
        "rc_wechat": "To WeChat campaigns", "rc_email": "To email campaigns",
    }),
]

ID_COLUMN = "customer_id"
BY_NAME = {f.name: f for f in FEATURES}
MODEL_FEATURES = [f for f in FEATURES if f.model]

# Columns in ds.customer_outcomes: validation only, never model inputs
OUTCOME_COLUMNS = ["converted", "conversion_rate", "med_days_to_close", "days_first_contact_to_win",
                   "n_transactions", "deal_value_gbp", "avg_discount_pct", "late_payment_rate",
                   "used_international_transfer", "med_first_reply_min"]
