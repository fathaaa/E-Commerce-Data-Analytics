import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

customers_df = pd.read_csv('customers_dataset.csv')
#geolocationdf = pd.read_csv('data\geolocation_dataset.csv')
itemorder_df = pd.read_csv('order_items_dataset.csv')
#paymentorder_df = pd.read_csv('data\order_payments_dataset.csv')
reviewsorder_df = pd.read_csv('order_reviews_dataset.csv')
datasetorder_df = pd.read_csv('orders_dataset.csv')
categoryproduct_df = pd.read_csv('product_category_name_translation.csv')
datasetproduct_df = pd.read_csv('products_dataset.csv')
sellers_df = pd.read_csv('sellers_dataset.csv')


datasetorder_df['order_delivered_customer_date'] = pd.to_datetime(datasetorder_df['order_delivered_customer_date'])
datetime_columns = ["order_delivered_customer_date"]
datasetorder_df.sort_values(by="order_delivered_customer_date", inplace=True)
datasetorder_df.reset_index(inplace=True)
###Side bar
for column in datetime_columns:
    datasetorder_df[column] = pd.to_datetime(datasetorder_df[column])

min_date = datasetorder_df["order_delivered_customer_date"].min()
max_date = datasetorder_df["order_delivered_customer_date"].max()

with st.sidebar:
    
    st.header("About Dashboard")
    st.markdown("""
                This dashboard is intended to see customer review scores, sales trends, average prices paid by customers, and what products sell best on the market. This visualization is expected to be a reference for company evaluation to be even better
               """)
    start_date = st.date_input(
        label='Start Date',
        min_value=datasetorder_df["order_delivered_customer_date"].min().date(),
        max_value=datasetorder_df["order_delivered_customer_date"].max().date(),
        value=datasetorder_df["order_delivered_customer_date"].min().date()
    )
    end_date = st.date_input(
        label='End Date',
        min_value=datasetorder_df["order_delivered_customer_date"].min().date(),
        max_value=datasetorder_df["order_delivered_customer_date"].max().date(),
        value=datasetorder_df["order_delivered_customer_date"].max().date()
    )

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    st.markdown("""



                
                Fatha Ariya Prasetya
               """)


###Data Wranngling Question 1 #####
itemreviews = pd.merge(
    left= reviewsorder_df,
    right=itemorder_df,
    how="inner",
    left_on= "order_id",
    right_on= "order_id"
)

itemreviews_seller_last = pd.merge(
    left=itemreviews,
    right=sellers_df,
    how="inner",
    left_on="seller_id",
    right_on="seller_id"
)

itemreviews_seller = pd.merge(
    left=itemreviews_seller_last,
    right=datasetorder_df,
    how="inner",
    left_on="order_id",
    right_on="order_id"
)

column_drop = ['review_comment_title','review_comment_message','review_answer_timestamp', 'order_item_id','shipping_limit_date','price','freight_value']
itemreviews_seller.drop(column_drop,axis=1,inplace=True)

###Data Wranngling Question 2 #####
order_and_customer = pd.merge(
    left= datasetorder_df,
    right= customers_df,
    how="inner",
    left_on="customer_id",
    right_on="customer_id"

)
order_and_customer.dropna(axis=0,inplace=True)
column_to_drop = ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','customer_unique_id']
order_and_customer.drop(column_to_drop,axis=1,inplace=True)

###Data Wranngling Question 3 #####
orderitem_full  = pd.merge(
    left= datasetorder_df,
    right= itemorder_df,
    how="inner",
    left_on="order_id",
    right_on="order_id"
)
order_product_last = pd.merge(
    left= orderitem_full,
    right= datasetproduct_df,
    how="inner",
    left_on="product_id",
    right_on="product_id"
)
order_product = pd.merge(
    left= order_product_last,
    right= customers_df,
    how="inner",
    left_on="customer_id",
    right_on="customer_id"
)
order_product.dropna(axis=0,subset=['product_category_name','order_delivered_customer_date'],inplace=True)
column_to_drop = ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_estimated_delivery_date','seller_id','shipping_limit_date','freight_value','product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm','customer_unique_id','customer_zip_code_prefix']
order_product.drop(column_to_drop,axis=1,inplace=True)





###Generate Function
def q1(start_datetime, end_datetime):
    
    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)

    
    itemreviews_seller_new = itemreviews_seller[
        (itemreviews_seller['order_delivered_customer_date'] >= start_datetime) & 
        (itemreviews_seller['order_delivered_customer_date'] <= end_datetime)
    ]

    state_mean_score = itemreviews_seller_new.groupby('seller_state')['review_score'].mean()
    state_mean_score_sorted = state_mean_score.sort_values(ascending=False)
    state_mean_score_sorted_df = pd.DataFrame(state_mean_score_sorted)

    seller_mean_score = itemreviews_seller_new.groupby(['seller_id','seller_state'])['review_score'].mean()
    seller_mean_score_sorted = seller_mean_score.sort_values(ascending=True)
    bad_seller = seller_mean_score_sorted[seller_mean_score_sorted < 3]
    bad_seller_grouped = bad_seller.groupby(level=1).count()
    bad_seller_grouped_sort = bad_seller_grouped.sort_values(ascending=False)
    bad_seller_grouped_sort_df = pd.DataFrame(bad_seller_grouped_sort)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    colors = ["#094e89","#2096be","#a1e0e7"]
    sns.barplot(x="seller_state",y="review_score", data = state_mean_score_sorted_df, palette = colors, ax=ax[0] )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Best State Review from Customer", loc="center", fontsize=15)
    ax[0].tick_params(axis ='y', labelsize=12)

    sns.barplot(x="seller_state",y="review_score", data = bad_seller_grouped_sort_df, palette = colors, ax=ax[1] )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Most State has Bad Sellers", loc="center", fontsize=15)
    ax[1].tick_params(axis ='y', labelsize=12)

    return fig

def q2(start_datetime, end_datetime):

    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)

    
    order_and_customer_new = order_and_customer[
        (itemreviews_seller['order_delivered_customer_date'] >= start_datetime) & 
        (itemreviews_seller['order_delivered_customer_date'] <= end_datetime)
    ]

    delivered_orders = order_and_customer_new[order_and_customer['order_status']=='delivered']
    delivered_byregion = delivered_orders.groupby('customer_state')['order_id'].count()
    delivered_byregion_sorted = delivered_byregion.sort_values(ascending=False)

    
    total_bymonth = order_and_customer_new.groupby(order_and_customer_new['order_delivered_customer_date'].dt.to_period('M'))['order_id'].count()
    total_bymonth_df = pd.DataFrame(total_bymonth)
    total_bymonth_df.rename(columns={ "order_id":"Total Order"}, inplace = True)
    end_month = end_datetime.strftime('%Y-%B')
    total_bymonth_df.index = total_bymonth_df.index.strftime('%Y-%B')
    
    total_bymonth_df_filtered = total_bymonth_df[total_bymonth_df.index <= end_month]
    

    total_byyear = order_and_customer_new.groupby(order_and_customer_new['order_delivered_customer_date'].dt.to_period('Y'))['order_id'].count()
    total_byyear_df = pd.DataFrame(total_byyear)
    total_byyear_df.rename(columns={ "order_id":"Total Order"}, inplace = True)
    end_year = end_datetime.strftime('%Y')
    total_byyear_df.index = total_byyear_df.index.strftime('%Y')
    total_byyear_df_filtered = total_byyear_df[total_byyear_df.index <= end_year]
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    ax[1].plot(total_bymonth_df_filtered.index,total_bymonth_df_filtered["Total Order"], marker='o', linewidth=2, color="#72BCD4" )
    ax[1].set_xticklabels(total_bymonth_df.index, rotation='vertical')
    ax[1].set_xlabel('Month')
    ax[1].set_ylabel('Total Order')
    ax[1].set_title('Total Orders per Month Chart')

    ax[0].plot(total_byyear_df_filtered.index,total_byyear_df_filtered["Total Order"], marker='o', linewidth=2, color="#72BCD4" )
    ax[0].set_xticklabels(total_byyear_df.index, rotation='horizontal')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Total Order')
    ax[0].set_title('Total Orders per Year Chart')

    return fig


def q3(start_datetime, end_datetime):
    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)
    
    order_product_new = order_product[
        (order_product['order_delivered_customer_date'] >= start_datetime) & 
        (order_product['order_delivered_customer_date'] <= end_datetime)
    ]
    state_mean_price = order_product_new.groupby('customer_state')['price'].mean()
    state_mean_price_sorted = state_mean_price.sort_values(ascending=False)
    state_mean_price_sorted_df = pd.DataFrame(state_mean_price_sorted)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,4))
    colors = ["#094e89","#2096be","#a1e0e7"]
    sns.barplot(x="customer_state",y="price", data = state_mean_price_sorted_df, palette = colors, ax=ax )
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_title("Average purchases by region", loc="center", fontsize=15)
    ax.tick_params(axis ='y', labelsize=12)

        
    
    return fig


def q3_1(start_datetime, end_datetime):
    start_datetime = pd.Timestamp(start_datetime)
    end_datetime = pd.Timestamp(end_datetime)
    
    order_product_new = order_product[
        (order_product['order_delivered_customer_date'] >= start_datetime) & 
        (order_product['order_delivered_customer_date'] <= end_datetime)
    ]

    state = ['PB', 'AL', 'AC']
    new_order_product = order_product_new[order_product['customer_state'].isin(state)]
    new_order_product_df = pd.DataFrame(new_order_product)
    state_top_categories = new_order_product.groupby('customer_state')['product_category_name'].value_counts().reset_index(name='count')
    state_top3_categories = state_top_categories.groupby('customer_state').apply(lambda x: x.nlargest(3, 'count')).reset_index(drop=True)

    fig, axes = plt.subplots(nrows=1, ncols=len(state), figsize=(16, 6))

    for i, st in enumerate(state):
        data_st = state_top3_categories[state_top3_categories['customer_state'] == st]
        ax = axes[i]
        sns.barplot(data=data_st, x='product_category_name', y='count', color='skyblue', ax=ax)
        ax.set_title('Top 3 Product Categories in ' + st)
        ax.set_ylabel('Count')
        ax.set_xlabel('Product Category')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticklabels(data_st['product_category_name'], rotation=25, ha='right')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    return fig


fig = q1(start_datetime, end_datetime)
fig2 = q2(start_datetime, end_datetime)
fig3 = q3(start_datetime, end_datetime)
fig3n = q3_1(start_datetime, end_datetime)




st.header('E-Commerce Data Analytics')

st.subheader("Customer's Review Score")


average_rating = itemreviews_seller["review_score"].mean()
st.metric("Average Rating Score (All Time)", value=average_rating)
 
st.pyplot(fig)
st.subheader("Sales Trends")
col1, col2 = st.columns(2)

with col1:
    total_orders = order_and_customer[order_and_customer['order_status'] == 'delivered']['order_id'].count()
    st.metric("Total orders", value=total_orders)
 
with col2:
    order_and_customer['order_delivered_customer_date'] = pd.to_datetime(order_and_customer['order_delivered_customer_date'])
    september_orders = order_and_customer[(order_and_customer['order_delivered_customer_date'].dt.year == 2018) &
                                      (order_and_customer['order_delivered_customer_date'].dt.month == 9)]

    total_orders_september = september_orders.shape[0]
    st.metric("Total Order in September 2018", value=total_orders_september)

st.pyplot(fig2)
st.subheader("Average Purchases by Region")
average_price = order_product["price"].mean()
st.metric("Average Purchases (All Time & State)", value=average_price)

st.pyplot(fig3)
st.subheader("The most sold product in the top 3 regions")
st.pyplot(fig3n)





 


    