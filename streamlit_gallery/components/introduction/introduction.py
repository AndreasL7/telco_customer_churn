import streamlit as st
import gc

def main():
    
    gc.enable()
    
    st.subheader("Welcome, Detective!")
    
    st.write("""
    In the rapidly evolving telecommunications landscape of Indonesia, a cutthroat competition amongst industry stalwarts shapes the market's trajectory. Spearheaded by key players such as Telkom Indonesia, Indosat Ooredoo, XL Axiata, Smartfren Telecom, and Tri Indonesia, the sector exhibits a dynamic interplay of market forces striving for a larger share of the connectivity domain​.
    """)
    
    image_url = "https://stl.tech/wp-content/uploads/2022/08/future-of-telcos-1.jpg"
    
    st.markdown(f'<a href="{image_url}"><img src="{image_url}" alt="description" width="700"/></a>', unsafe_allow_html=True)
    
    st.write("""
    The market is expected to grow from USD 13.52 billion in 2023 to USD 14.22 billion by 2028—intense competition is inevitable. Thus, companies need to be more strategic and intentional with customer acquisition and retention efforts. Amidst advancements such as the ushering of 5G technology in improving broader network infrastructure, customer churn remains a challenge for most, if not all companies in the space. As a result, optimizing customer satisfaction in this digital age is quintessential for failing to do so would risk long-term profitability and survivability.
    """)
    
    st.markdown("""<div style='text-align:center;position:relative;font-size:0;width:780px;'>
                        <img style='object-fit:cover;width:100%' alt='Telecom Market in Indonesia Size & Share Analysis - Growth Trends & Forecasts (2023 - 2028)' src='https://s3.mordorintelligence.com/indonesia-telecom-market/1667912322679_indonesia-telecom-market_Market_Summary.webp?embed=true'>
                        <a target='_blank' href='https://www.mordorintelligence.com/industry-reports/indonesia-telecom-market'>
                        <div style='position:absolute;bottom:0px;right:0px;color:white;background: rgba(0,0,0, 0.5);padding: 4px 6px;font-family:Arial;font-size:14px;'>
                        Image Source
                        </div>
                        </a>
                        </div>""", unsafe_allow_html=True)
        
    st.write("""
    
    Navigate to the **Prediction and Modelling** page to understand how our model works. 
    
    Grab your coffee and enjoy the investigation ahead! ☕️
    """)
    
    gc.collect()

if __name__ == "__main__":
    main()