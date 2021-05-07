import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("D:/Desktop/topology test.csv")
# fig = px.parallel_categories(df, color="val_epoch_loss",
#                               labels={"num_units1": "Num Units", "features": "Features",
#                                       "epoch_accuracy": "accuracy", "val_epoch_accuracy": "val accuracy",
#                                       "epoch_loss": "loss", "val_epoch_loss": "validation loss"},
#                               color_continuous_scale=px.colors.diverging.Tealrose,
#                              )

group_vars = df['features'].unique()
dfg = pd.DataFrame({'features':df['features'].unique()})
dfg['dummy'] = dfg.index
df = pd.merge(df, dfg, on = 'features', how='left')



fig = go.Figure(data=go.Parcoords(
    line=dict(color=df['val_epoch_loss'],
              colorscale='bluered',
              showscale=True,
              cmin=df['val_epoch_loss'].min(),
              cmax=df['val_epoch_loss'].max()),
    dimensions=list([
        dict(range=[df['num_units1'].min(), df['num_units1'].max()],
             label='Number of Neurons', values=df['num_units1']),
        dict(range=[0,df['dummy'].max()],
                       tickvals = dfg['dummy'], ticktext = dfg['features'],
                       label='Features', values=df['dummy']),
        dict(range=[df['epoch_accuracy'].min(), df['epoch_accuracy'].max()],
             label='Training Accuracy', values=df['epoch_accuracy']),
        dict(range=[df['val_epoch_accuracy'].min(), df['val_epoch_accuracy'].max()],
             label='Validation Accuracy', values=df['val_epoch_accuracy']),
        dict(range=[df['epoch_loss'].min(), df['epoch_loss'].max()],
             label='Training Loss', values=df['epoch_loss']),
        dict(range=[df['val_epoch_loss'].min(), df['val_epoch_loss'].max()],
             label='Validation Loss', values=df['val_epoch_loss']),

    ])
)
)
fig.show()
