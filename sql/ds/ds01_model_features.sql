-- ds01_model_features.sql
-- Model input features: one row per clean customer in ds.customer_features.
-- Built only from what a customer says (profile) and does (enquire, contact, visit, browse, respond).
-- Pipeline progress and results (stages, offers, applications, transactions, payments, agent reply times)
-- are excluded on purpose; they live in ds.customer_outcomes (ds02) and are used only to validate segments.
-- Shares are NULL when the customer has no activity of that kind (no denominator); preprocessing imputes them.
-- {{AS_OF}} (date, exclusive): only customers created and events that happened before it. Training uses the day
-- after the data end; the monthly refresh (d07) rebuilds the table as of each month start. Profile attributes
-- (budget, consent, ...) are the customer's current values: the source keeps no history of them.

CREATE SCHEMA IF NOT EXISTS ds;

-- Enquiries ------------------------------------------------------------------------------------
DROP TABLE IF EXISTS ds.feat_enquiry;
CREATE TABLE ds.feat_enquiry AS
SELECT customer_id,
       count(*)                                                        AS n_leads,
       count(DISTINCT date_trunc('month', created_date))               AS n_enquiry_months,
       avg((intent = 'Rent')::int)                                     AS share_rent_leads,
       avg((channel_group = 'Property portal')::int)                   AS ch_portal,
       avg((channel_group = 'Owned website')::int)                     AS ch_website,
       avg((channel_group = 'Paid search')::int)                       AS ch_paid_search,
       avg((channel IN ('Facebook', 'Instagram', 'TikTok'))::int)      AS ch_social,
       avg((channel = 'WeChat')::int)                                  AS ch_wechat,
       avg((channel_group = 'Email')::int)                             AS ch_email,
       avg((channel = 'Referral')::int)                                AS ch_referral,
       avg((channel = 'Overseas agent partner')::int)                  AS ch_overseas_partner,
       avg((channel = 'Relocation partner')::int)                      AS ch_relocation_partner,
       avg((channel_group = 'Branch')::int)                            AS ch_branch
FROM clean.leads
WHERE created_date < {{AS_OF}}::date
GROUP BY customer_id;

-- Contact with agents (interactions on the customer's leads) -----------------------------------
-- Topics tied to pipeline progress (viewing arrangements, offers, referencing) and generic property
-- questions are left out; the rest describe who the customer is.
DROP TABLE IF EXISTS ds.feat_contact;
CREATE TABLE ds.feat_contact AS
SELECT l.customer_id,
       count(*)                                                        AS n_interactions,
       avg((i.direction = 'Inbound')::int)                             AS share_inbound,
       avg((i.channel = 'Phone')::int)                                 AS ct_phone,
       avg((i.channel = 'Email')::int)                                 AS ct_email,
       avg((i.channel = 'WhatsApp')::int)                              AS ct_whatsapp,
       avg((i.channel = 'Live chat')::int)                             AS ct_live_chat,
       avg((i.channel = 'Video call')::int)                            AS ct_video_call,
       avg((i.channel = 'Branch visit')::int)                          AS ct_branch_visit,
       avg((i.topic = 'Mortgage and finance')::int)                    AS tp_mortgage,
       avg((i.topic = 'Stamp duty')::int)                              AS tp_stamp_duty,
       avg((i.topic = 'Rental yield')::int)                            AS tp_rental_yield,
       avg((i.topic = 'Chain and timescales')::int)                    AS tp_chain,
       avg((i.topic = 'Tenancy terms')::int)                           AS tp_tenancy_terms,
       avg((i.topic = 'Shared ownership')::int)                        AS tp_shared_ownership,
       avg((i.topic = 'Off-plan payment schedule')::int)               AS tp_offplan_schedule,
       avg((i.topic = 'School catchment')::int)                        AS tp_school_catchment,
       avg((i.topic = 'Retirement living')::int)                       AS tp_retirement,
       avg((i.topic = 'Visa and relocation')::int)                     AS tp_visa_relocation,
       avg((i.topic = 'Currency and transfers')::int)                  AS tp_currency
FROM clean.interactions i
JOIN clean.leads l USING (lead_id)
WHERE i.interaction_time < {{AS_OF}}::date
GROUP BY l.customer_id;

-- Viewings (how, not how they ended) ------------------------------------------------------------
DROP TABLE IF EXISTS ds.feat_visits;
CREATE TABLE ds.feat_visits AS
SELECT l.customer_id,
       count(*)                                                        AS n_site_visits,
       avg((v.visit_type = 'Virtual')::int)                            AS sv_virtual,
       avg((v.visit_type = 'Proxy (family or solicitor)')::int)        AS sv_proxy,
       avg((v.visit_type = 'Open house')::int)                         AS sv_open_house
FROM clean.site_visits v
JOIN clean.leads l USING (lead_id)
WHERE v.visit_date < {{AS_OF}}::date
GROUP BY l.customer_id;

-- Web sessions (timestamps are UK local time) ---------------------------------------------------
DROP TABLE IF EXISTS ds.feat_web;
CREATE TABLE ds.feat_web AS
SELECT customer_id,
       count(*)                                                        AS n_sessions,
       count(DISTINCT date_trunc('month', session_start))              AS n_active_months,
       avg((device = 'Mobile')::int)                                   AS dv_mobile,
       avg((device = 'Tablet')::int)                                   AS dv_tablet,
       avg((extract(hour FROM session_start) < 6)::int)                AS share_night_sessions,
       avg((ip_country <> 'GB')::int)                                  AS share_overseas_ip,
       avg((traffic_source = 'Property portal referral')::int)         AS tr_portal,
       avg((traffic_source = 'Organic search')::int)                   AS tr_organic_search,
       avg((traffic_source = 'Paid search')::int)                      AS tr_paid_search,
       avg((traffic_source IN ('Paid social', 'Organic social'))::int) AS tr_social,
       avg((traffic_source = 'Direct')::int)                           AS tr_direct,
       avg((traffic_source = 'Email')::int)                            AS tr_email,
       avg((traffic_source = 'Referral partner')::int)                 AS tr_partner,
       avg(pages_viewed)                                               AS avg_pages_per_session,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_sec)       AS med_session_sec
FROM clean.web_sessions
WHERE customer_id IS NOT NULL AND session_start < {{AS_OF}}::date
GROUP BY customer_id;

-- What and where they look at -------------------------------------------------------------------
DROP TABLE IF EXISTS ds.feat_views;
CREATE TABLE ds.feat_views AS
SELECT s.customer_id,
       count(*)                                                                    AS n_views,
       count(DISTINCT v.property_id)                                               AS n_properties_viewed,
       avg((v.event_type = 'Save')::int)                                           AS ev_save,
       avg((v.event_type = 'Virtual tour')::int)                                   AS ev_virtual_tour,
       avg((v.event_type = 'Share')::int)                                          AS ev_share,
       avg((v.event_type = 'Floorplan download')::int)                             AS ev_floorplan,
       avg((p.listing_purpose = 'Rent')::int)                                      AS share_let_views,
       avg((p.property_type = 'Flat')::int)                                        AS pt_flat,
       avg((p.property_type = 'Maisonette')::int)                                  AS pt_maisonette,
       avg((p.property_type = 'Terraced')::int)                                    AS pt_terraced,
       avg((p.property_type = 'Semi-detached')::int)                               AS pt_semi,
       avg((p.property_type = 'Detached')::int)                                    AS pt_detached,
       avg((p.property_type = 'Bungalow')::int)                                    AS pt_bungalow,
       avg(p.new_build::int)                                                       AS share_new_build,
       -- development_status is NULL for non-new-build homes; coalesce so avg() counts them as 0, not skips them
       avg(coalesce(p.development_status IN ('Off-plan', 'Under construction'), false)::int) AS share_offplan,
       avg((l.region = 'London')::int)                                             AS rg_london,
       avg((l.region IN ('South East', 'East of England'))::int)                   AS rg_south_east,
       avg((l.region IN ('North West', 'North East', 'Yorkshire and the Humber'))::int) AS rg_north,
       avg((l.region IN ('West Midlands', 'East Midlands'))::int)                  AS rg_midlands,
       avg((l.region IN ('South West', 'Wales'))::int)                             AS rg_south_west_wales,
       avg((l.region = 'Scotland')::int)                                           AS rg_scotland,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY p.asking_price_gbp) FILTER (WHERE p.listing_purpose = 'Sale') AS med_asking_viewed_gbp,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY p.rent_pcm_gbp)     FILTER (WHERE p.listing_purpose = 'Rent') AS med_rent_viewed_pcm_gbp,
       avg(l.avg_gross_yield_pct)                                                  AS avg_yield_viewed_pct
FROM clean.property_views v
JOIN clean.web_sessions s ON s.session_id = v.session_id
JOIN clean.properties p   ON p.property_id = v.property_id
JOIN clean.locations l    ON l.location_id = p.location_id
-- cut on the session start, as in feat_web, so a session and all its views stay together across midnight
WHERE s.customer_id IS NOT NULL AND s.session_start < {{AS_OF}}::date
GROUP BY s.customer_id;

-- Campaign responses ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ds.feat_campaign;
CREATE TABLE ds.feat_campaign AS
SELECT r.customer_id,
       count(*)                                                        AS n_campaign_responses,
       avg((r.response_type = 'Form submit')::int)                     AS rs_form_submit,
       avg((r.response_type IN ('Hide ad', 'Unsubscribe'))::int)       AS rs_negative,
       avg((c.channel = 'TikTok')::int)                                AS rc_tiktok,
       avg((c.channel = 'Meta')::int)                                  AS rc_meta,
       avg((c.channel = 'WeChat')::int)                                AS rc_wechat,
       avg((c.channel = 'Email')::int)                                 AS rc_email
FROM clean.campaign_responses r
JOIN clean.campaigns c USING (campaign_id)
WHERE r.response_time < {{AS_OF}}::date
GROUP BY r.customer_id;

-- One row per clean customer --------------------------------------------------------------------
DROP TABLE IF EXISTS ds.customer_features;
CREATE TABLE ds.customer_features AS
SELECT c.customer_id,
       -- profile
       c.customer_type, c.primary_intent, c.finance_type, c.buyer_position,
       c.age_band, c.occupation_sector, c.gender, c.country_iso2, c.is_overseas,
       coalesce(c.uk_region_of_residence, CASE WHEN c.is_overseas THEN 'Overseas' END) AS region_of_residence,
       c.budget_gbp, c.rent_budget_pcm_gbp,
       coalesce(c.budget_missing, c.rent_budget_missing)               AS budget_missing,
       c.preferred_currency <> 'GBP'                                   AS non_gbp_currency,
       c.consent_marketing,
       ({{AS_OF}}::date - 1) - c.first_contact_date                    AS tenure_days,
       c.probable_duplicate,
       -- activity presence
       e.customer_id IS NOT NULL                                       AS has_leads,
       w.customer_id IS NOT NULL                                       AS has_web,
       vw.customer_id IS NOT NULL                                      AS has_views,
       -- enquiry
       coalesce(e.n_leads, 0)                                          AS n_leads,
       coalesce(e.n_enquiry_months, 0)                                 AS n_enquiry_months,
       e.share_rent_leads, e.ch_portal, e.ch_website, e.ch_paid_search, e.ch_social, e.ch_wechat,
       e.ch_email, e.ch_referral, e.ch_overseas_partner, e.ch_relocation_partner, e.ch_branch,
       -- contact
       coalesce(ct.n_interactions, 0)                                  AS n_interactions,
       ct.share_inbound, ct.ct_phone, ct.ct_email, ct.ct_whatsapp, ct.ct_live_chat, ct.ct_video_call, ct.ct_branch_visit,
       ct.tp_mortgage, ct.tp_stamp_duty, ct.tp_rental_yield, ct.tp_chain, ct.tp_tenancy_terms, ct.tp_shared_ownership,
       ct.tp_offplan_schedule, ct.tp_school_catchment, ct.tp_retirement, ct.tp_visa_relocation, ct.tp_currency,
       -- viewings
       coalesce(sv.n_site_visits, 0)                                   AS n_site_visits,
       sv.sv_virtual, sv.sv_proxy, sv.sv_open_house,
       -- web
       coalesce(w.n_sessions, 0)                                       AS n_sessions,
       coalesce(w.n_active_months, 0)                                  AS n_active_months,
       w.dv_mobile, w.dv_tablet, w.share_night_sessions, w.share_overseas_ip,
       w.tr_portal, w.tr_organic_search, w.tr_paid_search, w.tr_social, w.tr_direct, w.tr_email, w.tr_partner,
       w.avg_pages_per_session, w.med_session_sec,
       -- property views
       coalesce(vw.n_views, 0)                                         AS n_views,
       coalesce(vw.n_properties_viewed, 0)                             AS n_properties_viewed,
       vw.n_views::numeric / nullif(w.n_sessions, 0)                   AS views_per_session,
       100.0 * coalesce(e.n_leads, 0) / nullif(vw.n_views, 0)          AS enquiries_per_100_views,
       vw.ev_save, vw.ev_virtual_tour, vw.ev_share, vw.ev_floorplan, vw.share_let_views,
       vw.pt_flat, vw.pt_maisonette, vw.pt_terraced, vw.pt_semi, vw.pt_detached, vw.pt_bungalow,
       vw.share_new_build, vw.share_offplan,
       vw.rg_london, vw.rg_south_east, vw.rg_north, vw.rg_midlands, vw.rg_south_west_wales, vw.rg_scotland,
       vw.med_asking_viewed_gbp, vw.med_rent_viewed_pcm_gbp, vw.avg_yield_viewed_pct,
       -- campaigns
       coalesce(cr.n_campaign_responses, 0)                            AS n_campaign_responses,
       cr.rs_form_submit, cr.rs_negative, cr.rc_tiktok, cr.rc_meta, cr.rc_wechat, cr.rc_email
FROM clean.customers c
LEFT JOIN ds.feat_enquiry  e  USING (customer_id)
LEFT JOIN ds.feat_contact  ct USING (customer_id)
LEFT JOIN ds.feat_visits   sv USING (customer_id)
LEFT JOIN ds.feat_web      w  USING (customer_id)
LEFT JOIN ds.feat_views    vw USING (customer_id)
LEFT JOIN ds.feat_campaign cr USING (customer_id)
WHERE c.created_at < {{AS_OF}}::date;

ALTER TABLE ds.customer_features ADD PRIMARY KEY (customer_id);
