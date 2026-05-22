def forecast_revenue(current_revenue,growth_rate,forecast_years=5):
    forecasted_revenues=[]
    for year in range(1,forecast_years+1):
      future_revenue=(current_revenue*(1+growth_rate)**year)
      forecasted_revenues.append(future_revenue)
    return forecasted_revenues

def forecast_fcf(revenue_forecasts,fcf_margin=0.20):
   fcf_forecasts=[]
   for revenue in revenue_forecasts:
      fcf=revenue*fcf_margin
      fcf_forecasts.append(fcf)
   return fcf_forecasts

def discount_cash_flow(cash_flows,discount_rate=0.10):
   discounted_flows=[]
   for year,cash_flow in enumerate(cash_flows,start=1):
      present_value=(cash_flow/(1+discount_rate)**year)
      discounted_flows.append(present_value)
   return discounted_flows
   

if __name__=="__main__":
  current_revenue=100_000_000_000
  growth_rate=0.10

  forecasts=forecast_revenue(
    current_revenue,
    growth_rate,
    forecast_years=5
  )


  print("\n===REVENUE FORECASTS===")
  for i, revenue in enumerate(forecasts,start=1):
    print(f"Year {i}:${revenue:,.2f}")
  

  fcf_forecasts=forecast_fcf(forecasts,fcf_margin=0.20)
  print("\n===FCF FORECASTS===")
  for i,fcf in enumerate(fcf_forecasts,start=1):
     print(f"Year{i}:${fcf:,.2f}")

  discounted_fcfs=discount_cash_flow(fcf_forecasts,discount_rate=0.12)
  print("\n===PRESENT VALUE===")
  for i,pv in enumerate(discounted_fcfs,start=1):
     print(f"Year{i}:${pv:,.2f}")
  

