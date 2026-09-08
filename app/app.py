import streamlit as st
import pandas as pd
import joblib
import re
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medicine Information & Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = r"C:\big data\medical\medicine_treatment_rf_model.pkl"
DATA_PATH = r"C:\big data\medical\medicine_summary.csv"
BACKGROUND_IMAGE = r"C:\big data\medical\medicin bg.png"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

@st.cache_data
def load_background_image():

    with open(BACKGROUND_IMAGE, "rb") as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode()


# =========================================================
# LOAD EVERYTHING
# =========================================================

rf_model = load_model()

medicine_data = load_data()

background_image = load_background_image()


# =========================================================
# SESSION STATE
# =========================================================

if "search_done" not in st.session_state:
    st.session_state.search_done = False

if "search_result" not in st.session_state:
    st.session_state.search_result = None

if "search_type" not in st.session_state:
    st.session_state.search_type = None

if "search_title" not in st.session_state:
    st.session_state.search_title = ""


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
<style>

/* =====================================================
   PAGE BACKGROUND
===================================================== */

.stApp {{
    background-image:
        url("data:image/png;base64,{background_image}") !important;

    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}


/* =====================================================
   MAIN WIDTH
===================================================== */

.block-container {{
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 50px;
}}


/* =====================================================
   GLOBAL TEXT
===================================================== */

.stApp p {{
    color: #18324b !important;
}}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {{
    color: #123b70 !important;
}}


/* =====================================================
   HEADER
===================================================== */

.title-container {{
    text-align: center;
    margin-bottom: 25px;
}}

.title-container h1 {{
    color: #123b70 !important;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}}

.title-container p {{
    color: #38566e !important;
    font-size: 17px;
}}


/* =====================================================
   SELECT BOX LABEL
===================================================== */

[data-testid="stSelectbox"] label {{
    color: #244b68 !important;
    font-weight: 700 !important;
}}


/* =====================================================
   SELECT BOX
===================================================== */

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: #252731 !important;
    border-radius: 10px !important;
}}

[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    color: #ffffff !important;
}}

[data-testid="stSelectbox"] svg {{
    fill: #ffffff !important;
}}


/* =====================================================
   BUTTON
===================================================== */

.stButton > button {{
    width: 100%;
    min-height: 50px;
    border-radius: 12px;
    border: none;

    background: linear-gradient(
        90deg,
        #2188e6,
        #36a4eb
    );

    color: #ffffff !important;

    font-size: 17px;
    font-weight: 700;
}}

.stButton > button p {{
    color: #ffffff !important;
}}

.stButton > button:hover {{
    background: linear-gradient(
        90deg,
        #1976d2,
        #238bd7
    );

    color: #ffffff !important;
}}


/* =====================================================
   NATIVE STREAMLIT CONTAINERS
===================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(255, 255, 255, 0.94) !important;

    border: 1px solid #d8e8ef !important;

    border-radius: 22px !important;

    box-shadow:
        0 8px 28px rgba(30, 80, 110, 0.12) !important;
}}


/* =====================================================
   CONTAINER HEADINGS
===================================================== */

[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] h4 {{
    color: #123b70 !important;
}}


/* =====================================================
   DETAIL BOX
===================================================== */

.detail-box {{
    background: #f8fcff;

    padding: 15px;

    border-radius: 12px;

    border: 1px solid #d8e8ef;

    margin-bottom: 10px;

    color: #18324b !important;
}}

.detail-box b {{
    color: #123b70 !important;
}}


/* =====================================================
   RED SIDE EFFECT BOX
===================================================== */

.side-effect-box {{
    background: #ff4d4d !important;

    border: 1px solid #d62f2f !important;

    border-radius: 12px;

    padding: 16px;

    margin-top: 10px;

    margin-bottom: 10px;

    color: #000000 !important;
}}

.side-effect-title {{
    color: #000000 !important;

    font-weight: 800;

    font-size: 16px;

    margin-bottom: 8px;
}}

.side-effect-text {{
    color: #000000 !important;

    font-size: 15px;

    line-height: 1.6;
}}


/* =====================================================
   DIALOG
===================================================== */

[data-testid="stDialog"] h1,
[data-testid="stDialog"] h2,
[data-testid="stDialog"] h3,
[data-testid="stDialog"] h4 {{
    color: #123b70 !important;
}}

[data-testid="stDialog"] p {{
    color: #18324b !important;
}}


/* =====================================================
   EXPANDER
===================================================== */

[data-testid="stExpander"] summary p {{
    color: #123b70 !important;

    font-weight: 700 !important;
}}


/* =====================================================
   FOOTER
===================================================== */

.footer-text {{
    text-align: center;

    color: #38566e !important;
}}

.footer-text b {{
    color: #123b70 !important;
}}


/* =====================================================
   PRESCRIPTION WARNING
===================================================== */

.prescription-warning {{
    text-align: center;

    color: #d00000 !important;

    font-size: 18px;

    font-weight: 800;

    margin-top: 15px;

    margin-bottom: 10px;
}}


/* =====================================================
   HIDE STREAMLIT MENU / FOOTER
===================================================== */

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# =========================================================
# EXTRACT SIDE EFFECTS
# =========================================================

def extract_individual_side_effects(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Split by | OR comma
    parts = re.split(r"[|,]", text)

    effects = []

    for part in parts:

        effect = part.strip()

        if not effect:
            continue

        if effect.lower() in [
            "none",
            "none reported",
            "null",
            "nan",
            "not reported"
        ]:
            continue

        effects.append(effect)

    return effects


# =========================================================
# GET MAXIMUM 5 SIDE EFFECTS
# =========================================================

def get_top_5_side_effects(value):

    effects = extract_individual_side_effects(value)

    if not effects:
        return []

    counts = {}

    for effect in effects:

        effect_key = effect.lower()

        if effect_key not in counts:

            counts[effect_key] = {
                "name": effect,
                "count": 0
            }

        counts[effect_key]["count"] += 1

    sorted_effects = sorted(
        counts.values(),
        key=lambda x: x["count"],
        reverse=True
    )

    return [
        item["name"]
        for item in sorted_effects[:5]
    ]


# =========================================================
# FORMAT SIDE EFFECTS
# =========================================================

def format_side_effects(value):

    effects = get_top_5_side_effects(value)

    if not effects:

        return "No side effects reported"

    return ", ".join(effects)


# =========================================================
# FORMAT PRICE
# =========================================================

def format_price(value):

    if pd.isna(value):
        return "Not listed"

    text = str(value).strip()

    if not text:
        return "Not listed"

    values = text.split("|")

    valid_prices = []

    for item in values:

        cleaned = (
            item
            .strip()
            .replace("₹", "")
            .replace(",", "")
        )

        try:

            price = float(cleaned)

            if price < 11:

                formatted = f"₹{price:,.2f}"

                if formatted not in valid_prices:

                    valid_prices.append(formatted)

        except ValueError:

            continue

    if valid_prices:

        return " | ".join(valid_prices)

    return "Not listed"


# =========================================================
# FORMAT PURITY
# =========================================================

def format_purity(value):

    if pd.isna(value):
        return "Not listed"

    text = str(value).strip()

    if not text:
        return "Not listed"

    values = text.split("|")

    allowed_purity = {
        95.0,
        96.0,
        97.0,
        98.0
    }

    valid_purity = []

    for item in values:

        cleaned = (
            item
            .strip()
            .replace("%", "")
        )

        try:

            purity = float(cleaned)

            if purity in allowed_purity:

                formatted = f"{purity:.1f}%"

                if formatted not in valid_purity:

                    valid_purity.append(formatted)

        except ValueError:

            continue

    if valid_purity:

        return " | ".join(valid_purity)

    return "Not listed"


# =========================================================
# DYNAMIC TABLE HEIGHT
# =========================================================

def dynamic_table_height(number_of_rows):

    row_height = 35

    header_height = 38

    height = (
        header_height
        + number_of_rows * row_height
    )

    return min(
        max(height, 90),
        450
    )


# =========================================================
# FULL RESULT POPUP
# =========================================================

@st.dialog(
    "🎯 Your Result",
    width="large"
)
def show_result_popup(result, search_type):

    st.write(
        "Complete medicine information matching your search."
    )


    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    full_result = (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )


    # =====================================================
    # COMPLETE MEDICINE DETAILS
    # =====================================================

    st.subheader(
        "💊 Complete Medicine Details"
    )


    detail_columns = [
        "Disease",
        "Treatment",
        "Tablet Name",
        "Dosage",
        "Time Taken",
        "Effect Time Duration",
        "Which Age",
        "Company",
        "Purity",
        "Side Effects",
        "Price"
    ]


    available_columns = [
        column
        for column in detail_columns
        if column in full_result.columns
    ]


    display_result = full_result[
        available_columns
    ].copy()


    # =====================================================
    # FORMAT COLUMNS
    # =====================================================

    if "Side Effects" in display_result.columns:

        display_result["Side Effects"] = (
            display_result["Side Effects"]
            .apply(format_side_effects)
        )


    if "Price" in display_result.columns:

        display_result["Price"] = (
            display_result["Price"]
            .apply(format_price)
        )


    if "Purity" in display_result.columns:

        display_result["Purity"] = (
            display_result["Purity"]
            .apply(format_purity)
        )


    # =====================================================
    # REMOVE EMPTY ROWS
    # =====================================================

    display_result = (
        display_result
        .dropna(how="all")
        .reset_index(drop=True)
    )


    # =====================================================
    # DISPLAY TABLE
    # =====================================================

    if not display_result.empty:

        st.dataframe(
            display_result,
            use_container_width=True,
            hide_index=True,
            height=dynamic_table_height(
                len(display_result)
            )
        )


    st.divider()


    # =====================================================
    # MEDICINE-WISE DETAILS
    # =====================================================

    st.subheader(
        "🔎 Medicine-wise Details"
    )


    for index, medicine in full_result.iterrows():

        tablet_name = clean_text(
            medicine.get(
                "Tablet Name",
                "Medicine"
            )
        )

        treatment_name = clean_text(
            medicine.get(
                "Treatment",
                ""
            )
        )


        with st.expander(
            f"💊 {tablet_name} — {treatment_name}"
        ):


            col1, col2 = st.columns(2)


            # =============================================
            # LEFT COLUMN
            # =============================================

            with col1:

                st.markdown(
                    f"""
<div class="detail-box">
🩺 <b>Disease</b><br>
{clean_text(medicine.get("Disease", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
📋 <b>Treatment</b><br>
{clean_text(medicine.get("Treatment", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
💊 <b>Tablet Name</b><br>
{clean_text(medicine.get("Tablet Name", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
💉 <b>Dosage</b><br>
{clean_text(medicine.get("Dosage", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
⏰ <b>Time Taken</b><br>
{clean_text(medicine.get("Time Taken", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
⏱️ <b>Effect Time Duration</b><br>
{clean_text(medicine.get("Effect Time Duration", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


            # =============================================
            # RIGHT COLUMN
            # =============================================

            with col2:

                st.markdown(
                    f"""
<div class="detail-box">
👤 <b>Age Group</b><br>
{clean_text(medicine.get("Which Age", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
<div class="detail-box">
🏢 <b>Company</b><br>
{clean_text(medicine.get("Company", "N/A"))}
</div>
""",
                    unsafe_allow_html=True
                )


                # =========================================
                # PURITY
                # =========================================

                purity = format_purity(
                    medicine.get(
                        "Purity",
                        ""
                    )
                )


                st.markdown(
                    f"""
<div class="detail-box">
🧪 <b>Purity</b><br>
{purity}
</div>
""",
                    unsafe_allow_html=True
                )


                # =========================================
                # PRICE
                # =========================================

                price = format_price(
                    medicine.get(
                        "Price",
                        ""
                    )
                )


                st.markdown(
                    f"""
<div class="detail-box">
💰 <b>Price</b><br>
{price}
</div>
""",
                    unsafe_allow_html=True
                )


                # =========================================
                # SIDE EFFECTS
                # =========================================

                side_effects = get_top_5_side_effects(
                    medicine.get(
                        "Side Effects",
                        ""
                    )
                )


                if side_effects:

                    side_effect_text = ", ".join(
                        side_effects[:5]
                    )

                else:

                    side_effect_text = (
                        "No side effects reported"
                    )


                st.markdown(
                    f"""
<div class="side-effect-box">

<div class="side-effect-title">
⚠️ Side Effects
</div>

<div class="side-effect-text">
{side_effect_text}
</div>

</div>
""",
                    unsafe_allow_html=True
                )


    st.divider()


    # =====================================================
    # SAFETY MESSAGE
    # =====================================================

    st.caption(
        "⚠️ This information is retrieved from the project "
        "dataset for informational purposes only. It is not a "
        "medical prescription or substitute for professional "
        "medical advice."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="title-container">

<h1>
💊 Medicine Information & Recommendation System
</h1>

<p>
🔎 Find medicine information using Disease,
Tablet Name and Treatment.
</p>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# MEDICINE SEARCH
# =========================================================

with st.container(border=True):

    st.subheader(
        "🔍 Medicine Search"
    )


    st.write(
        "Select one, two, or all three fields. "
        "Leave any field empty when you do not need it."
    )


    # =====================================================
    # INPUT COLUMNS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    # =====================================================
    # DISEASE
    # =====================================================

    with col1:

        disease_options = [
            "Select Disease"
        ] + sorted(
            medicine_data["Disease"]
            .dropna()
            .unique()
            .tolist()
        )


        disease = st.selectbox(
            "🩺 Disease",
            disease_options,
            key="disease_input"
        )


    # =====================================================
    # TABLET
    # =====================================================

    with col2:

        tablet_options = [
            "Select Tablet"
        ] + sorted(
            medicine_data["Tablet Name"]
            .dropna()
            .unique()
            .tolist()
        )


        tablet = st.selectbox(
            "💊 Tablet Name",
            tablet_options,
            key="tablet_input"
        )


    # =====================================================
    # TREATMENT
    # =====================================================

    with col3:

        treatment_options = [
            "Select Treatment"
        ] + sorted(
            medicine_data["Treatment"]
            .dropna()
            .unique()
            .tolist()
        )


        treatment = st.selectbox(
            "📋 Treatment",
            treatment_options,
            key="treatment_input"
        )


    # =====================================================
    # CONVERT DEFAULT VALUES
    # =====================================================

    if disease == "Select Disease":

        disease = None


    if tablet == "Select Tablet":

        tablet = None


    if treatment == "Select Treatment":

        treatment = None


    # =====================================================
    # SEARCH BUTTON
    # =====================================================

    st.write("")


    search_button = st.button(
        "🔎 Find Information",
        use_container_width=True
    )


# =========================================================
# SEARCH LOGIC
# =========================================================

if search_button:


    # =====================================================
    # RESET OLD RESULT
    # =====================================================

    st.session_state.search_done = False

    st.session_state.search_result = None

    st.session_state.search_type = None

    st.session_state.search_title = ""


    # =====================================================
    # CHECK EMPTY SEARCH
    # =====================================================

    if (
        disease is None
        and tablet is None
        and treatment is None
    ):

        st.warning(
            "⚠️ Please select at least one field to search."
        )


    else:


        # =================================================
        # START WITH COMPLETE DATA
        # =================================================

        result = medicine_data.copy()


        # =================================================
        # FILTER DISEASE
        # =================================================

        if disease is not None:

            result = result[
                result["Disease"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                disease.strip().lower()
            ]


        # =================================================
        # FILTER TABLET
        # =================================================

        if tablet is not None:

            result = result[
                result["Tablet Name"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                tablet.strip().lower()
            ]


        # =================================================
        # FILTER TREATMENT
        # =================================================

        if treatment is not None:

            result = result[
                result["Treatment"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                treatment.strip().lower()
            ]


        # =================================================
        # CHECK RESULT
        # =================================================

        if result.empty:

            st.error(
                "❌ No matching information was found."
            )


        else:


            # =================================================
            # REMOVE EXACT DUPLICATES
            # =================================================

            result = (
                result
                .drop_duplicates()
                .reset_index(drop=True)
            )


            # =================================================
            # SEARCH TYPE
            # =================================================

            if (
                disease is not None
                and tablet is None
                and treatment is None
            ):

                search_type = "disease_only"

                search_title = (
                    f"Medicines for {disease}"
                )


            elif (
                disease is None
                and tablet is not None
                and treatment is None
            ):

                search_type = "tablet_only"

                search_title = (
                    f"Information for {tablet}"
                )


            elif (
                disease is None
                and tablet is None
                and treatment is not None
            ):

                search_type = "treatment_only"

                search_title = (
                    f"Information for {treatment}"
                )


            elif (
                disease is not None
                and tablet is not None
                and treatment is None
            ):

                search_type = "disease_tablet"

                search_title = (
                    "Corresponding Treatment"
                )


            elif (
                disease is not None
                and tablet is None
                and treatment is not None
            ):

                search_type = "disease_treatment"

                search_title = (
                    "Corresponding Tablet"
                )


            elif (
                disease is None
                and tablet is not None
                and treatment is not None
            ):

                search_type = "tablet_treatment"

                search_title = (
                    "Corresponding Disease"
                )


            else:

                search_type = "all"

                search_title = (
                    "Complete Medicine Information"
                )


            # =================================================
            # SAVE RESULT
            # =================================================

            st.session_state.search_done = True

            st.session_state.search_result = result

            st.session_state.search_type = search_type

            st.session_state.search_title = search_title


# =========================================================
# DISPLAY SEARCH RESULT
# =========================================================

if (
    st.session_state.search_done
    and st.session_state.search_result is not None
):


    result = st.session_state.search_result

    search_type = st.session_state.search_type


    # =====================================================
    # RESULT CONTAINER
    # =====================================================

    with st.container(border=True):


        st.subheader(
            "🎯 Your Result"
        )


        st.write(
            "Matching information from the medicine dataset."
        )


        # =================================================
        # DISEASE ONLY
        # =================================================

        if search_type == "disease_only":


            st.subheader(
                "💊 Corresponding Medicines"
            )


            output = (
                result
                .groupby("Treatment")["Tablet Name"]
                .apply(
                    lambda x:
                        ", ".join(
                            sorted(
                                x.dropna()
                                .astype(str)
                                .unique()
                            )
                        )
                )
                .reset_index()
            )


            output.columns = [
                "Treatment",
                "Tablet Name"
            ]


            output = (
                output
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # TABLET ONLY
        # =================================================

        elif search_type == "tablet_only":


            st.subheader(
                "💊 Corresponding Information"
            )


            output = (
                result[
                    [
                        "Disease",
                        "Treatment"
                    ]
                ]
                .drop_duplicates()
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # TREATMENT ONLY
        # =================================================

        elif search_type == "treatment_only":


            st.subheader(
                "📋 Corresponding Information"
            )


            output = (
                result
                .groupby("Disease")["Tablet Name"]
                .apply(
                    lambda x:
                        ", ".join(
                            sorted(
                                x.dropna()
                                .astype(str)
                                .unique()
                            )
                        )
                )
                .reset_index()
            )


            output.columns = [
                "Disease",
                "Tablet Name"
            ]


            output = (
                output
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # DISEASE + TABLET
        # =================================================

        elif search_type == "disease_tablet":


            st.subheader(
                "📋 Corresponding Treatment"
            )


            output = (
                result[
                    ["Treatment"]
                ]
                .drop_duplicates()
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # DISEASE + TREATMENT
        # =================================================

        elif search_type == "disease_treatment":


            st.subheader(
                "💊 Corresponding Tablet"
            )


            output = (
                result[
                    ["Tablet Name"]
                ]
                .drop_duplicates()
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # TABLET + TREATMENT
        # =================================================

        elif search_type == "tablet_treatment":


            st.subheader(
                "🦠 Corresponding Disease"
            )


            output = (
                result[
                    ["Disease"]
                ]
                .drop_duplicates()
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(output)
                )
            )


        # =================================================
        # ALL THREE
        # =================================================

        elif search_type == "all":


            st.subheader(
                "💊 Complete Medicine Information"
            )


            preview_columns = [
                "Disease",
                "Treatment",
                "Tablet Name"
            ]


            preview_columns = [
                column
                for column in preview_columns
                if column in result.columns
            ]


            preview = (
                result[
                    preview_columns
                ]
                .drop_duplicates()
                .dropna(how="all")
                .reset_index(drop=True)
            )


            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True,
                height=dynamic_table_height(
                    len(preview)
                )
            )


        # =================================================
        # VIEW FULL RESULT
        # =================================================

        st.write("")


        view_full_result = st.button(
            "👁️ View Full Result",
            use_container_width=True,
            key="view_full_result_button"
        )


        if view_full_result:

            show_result_popup(
                st.session_state.search_result,
                st.session_state.search_type
            )


# =========================================================
# BOTTOM INFORMATION SECTION
# =========================================================

st.write("")

st.write("")

st.divider()

st.write("")


col1, col2 = st.columns(2)


# =========================================================
# HOW TO USE
# =========================================================

with col1:

    with st.container(border=True):


        st.subheader(
            "💡 How to use?"
        )


        st.write(
            "You can search using any combination "
            "of the three fields:"
        )


        st.write(
            "• **Disease only** → Find corresponding "
            "Tablets and Treatments"
        )


        st.write(
            "• **Tablet only** → Find corresponding "
            "Diseases and Treatments"
        )


        st.write(
            "• **Treatment only** → Find corresponding "
            "Diseases and Tablets"
        )


        st.write(
            "• **Any two fields** → Find the missing "
            "information"
        )


        st.write(
            "• **All three fields** → View complete "
            "medicine details"
        )


# =========================================================
# WHY USE THIS SYSTEM
# =========================================================

with col2:

    with st.container(border=True):


        st.subheader(
            "✅ Why use this system?"
        )


        st.write(
            "• Quick access to medicine information"
        )


        st.write(
            "• Data-driven medicine relationships"
        )


        st.write(
            "• Easy Disease, Tablet and Treatment search"
        )


        st.write(
            "• Complete medicine details when available"
        )


        st.write(
            "• Machine learning supported treatment prediction"
        )


        st.write(
            "• Simple and user-friendly interface"
        )


# =========================================================
# FOOTER
# =========================================================

st.write("")


st.markdown(
    """
<div class="footer-text">

💊 <b>Medicine Information & Recommendation System</b>

<br><br>

Data-driven medicine information for better awareness.

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# PRESCRIPTION WARNING
# =========================================================

st.markdown(
    """
<div class="prescription-warning">

⚠️ Don't Buy any Tablets without PRESCRIPTION

</div>
""",
    unsafe_allow_html=True
)