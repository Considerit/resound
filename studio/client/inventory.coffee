
dom.INVENTORY = ->
  song = @props.song

  manifest = retrieve("/manifest/#{song}")

  channels = retrieve('/channels').channels
  return DIV null if !channels

  if !manifest.manifest
    return SPAN null

  reactions = Object.values(manifest.manifest.reactions)

  @local.sort ?= 'views'

  if @local.sort == 'views'
    reactions.sort (a,b) -> b.views - a.views
  else if @local.sort == 'alphabetical'
    reactions.sort (a,b) -> alphabetical_compare(a.reactor, b.reactor)
  else # channel subscriptions
    reactions.sort( (b,a) -> channels[a.channelId].subscriberCount - channels[b.channelId].subscriberCount  )

  channel_recommendation_filters = ['include', 'exclude', 'eligible', '']
  if !@local.channel_recommendation_filters
    @local.channel_recommendation_filters ?= {}
    for filter in channel_recommendation_filters
      @local.channel_recommendation_filters[filter] = true

  included_filters = [true, false]
  if !@local.included_filters
    @local.included_filters ?= {}
    for filter in included_filters
      @local.included_filters[filter] = true

  DIV null,
    DIV null, 
      SPAN null,
        'sort: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for sort in ['views', 'subscribers', 'alphabetical']
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(sort) => => 
                @local.sort = sort
                save @local

              style: 
                backgroundColor: if @local.sort == sort then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              sort

    DIV null, 
      SPAN null,
        'channel recommendation: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for filter in channel_recommendation_filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.channel_recommendation_filters[filter] = !@local.channel_recommendation_filters[filter]
                save @local

              style: 
                backgroundColor: if @local.channel_recommendation_filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              if filter.length > 0
                filter
              else
                '<unset>'

    DIV null, 
      SPAN null,
        'included: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for filter in included_filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.included_filters[filter] = !@local.included_filters[filter]
                save @local

              style: 
                backgroundColor: if @local.included_filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              "#{filter}"

    DIV null,

      INPUT 
        key: 'filter'
        type: 'text'
        defaultValue: @local.filter_channels or "" 
        onChange: (e) => 
          @local.filter_channels = e.target.value
          save @local


    UL 
      style:
        listStyle: 'none'

      for reaction in reactions
        if @local.channel_recommendation_filters[ channels[reaction.channelId]?.auto or '' ] && @local.included_filters[ reaction.download or false ]
          if !@local.filter_channels || reaction.reactor.toLowerCase().indexOf(@local.filter_channels.toLowerCase()) > -1
            MANIFEST_ITEM
              song: song
              reaction: reaction

dom.MANIFEST_ITEM = ->
  reaction = @props.reaction
  song = @props.song

  key = "/reaction/#{reaction.id}"
  retrieve(key)

  explicit = !!reaction.explicit
  included = !!reaction.download

  if included
    color = if explicit then '#91f2ab' else '#b9ddb2'
  else
    color = if explicit then '#ddb2b5' else '#fafafa'

  @local.show_iframe = @local.show_iframe or @local.in_viewport

  channels = retrieve('/channels').channels
  my_channel = channels[reaction.channelId]

  channel_warning = my_channel?.auto == 'exclude'

  LI 
    'data-receive-viewport-visibility-updates': 1
    "data-component": @local.key

    style: 
      backgroundColor: color
      padding: '18px'
      margin: 9
      borderRadius: 8


    if channel_warning
      DIV 
        style:
          backgroundColor: 'red'
          color: 'white'
          fontSize: 20
          display: 'inline-block'
        'CHANNEL EXCLUDED'

    DIV
      style: 
        fontSize: 14
      "#{reaction.views} views"

    DIV
      style: 
        fontSize: 14
      "#{reaction.release_date}"

    H3 null,
      reaction.reactor

    DIV
      style: 
        fontSize: 16
        marginBottom: 8
      dangerouslySetInnerHTML: __html: reaction.title

    DIV null,
      BUTTON
        onClick: ->
          reaction.explicit = true
          reaction.download = true
          save {
            key: key,
            song: song,
            reaction: reaction
          }

        style: 
          fontSize: 18
          backgroundColor: 'green'
          margin: 9
          borderRadius: 8
          borderColor: 'transparent'
          color: 'white'
          fontWeight: 'bold'
          padding: '8px 12px'          
        "Pass"

      BUTTON 
        onClick: ->
          reaction.explicit = true
          reaction.download = false
          save {
            key: key,
            song: song,
            reaction: reaction
          }    

        style: 
          fontSize: 18
          backgroundColor: 'red'
          margin: 9          
          borderRadius: 8
          borderColor: 'transparent'
          color: 'white'
          fontWeight: 'bold'
          padding: '8px 12px'
        "Fail"    
      
    if @local.show_iframe
      IFRAME
        width: "560" 
        height: "315" 
        src: "https://www.youtube.com/embed/#{reaction.id}" 
        frameborder: "0" 
        allow: "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
        allowfullscreen: true
    else
      DIV
        style:
          width: 560
          height: 315 

