
dom.CHANNELS = ->
  channels = retrieve('/channels').channels
  return DIV null if !channels

  all = Object.values(channels)

  @local.sort ?= 'activity'

  if @local.sort == 'inclusions'
    all.sort (b,a) ->
      ai = (a.included_in or []).length
      bi = (b.included_in or []).length

      if ai != bi
        ai - bi
      else
        ai = (a.mentioned_in or []).length
        bi = (b.mentioned_in or []).length
        ai - bi

  else if @local.sort == 'alphabetical'
    all.sort (b,a) -> alphabetical_compare(a.reactor, b.reactor)
  else
    all.sort( (b,a) -> (a.viewCount or 1) * (a.subscriberCount or 1) - (b.viewCount or 1) * (b.subscriberCount or 1)  )

  filters = ['include', 'exclude', 'eligible', '']
  if !@local.filters
    @local.filters ?= {}
    for filter in filters
      @local.filters[filter] = true


  songs = retrieve('/songs').songs

  return SPAN null if !songs

  mentioned_in_options = [' '].concat(songs.slice())

  @local.mentioned_in ?= ' '
  @local.included_in ?= ' '

  DIV null,

    DIV null, 
      SPAN null,
        'sort: '
      UL 
        style: 
          display: 'inline'
          margin: '8px 0'
          padding: 0

        for sort in ['activity', 'inclusions', 'alphabetical']
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

        for filter in filters
          LI
            style:
              listStyle: 'none'
              display: 'inline'

            BUTTON
              onClick: do(filter) => => 
                @local.filters[filter] = !@local.filters[filter]
                save @local

              style: 
                backgroundColor: if @local.filters[filter] then '#f2f2f2' else 'transparent'
                padding: '4px 8px'
                border: 'none'
                cursor: 'pointer'

              if filter.length > 0
                filter
              else
                '<unset>'

    DIV null, 
      SPAN null,
        'mentioned in: '

      SELECT 
        style: 
          fontSize: 18
        value: @local.mentioned_in
        onChange: (e) => 
          @local.mentioned_in = e.target.value
          save @local 

        for n,idx in mentioned_in_options
          OPTION 
            value: n
            style: {}
            n

    DIV null, 
      SPAN null,
        'included in: '

      SELECT 
        style: 
          fontSize: 18
        value: @local.included_in
        onChange: (e) => 
          @local.included_in = e.target.value
          save @local 

        for n,idx in mentioned_in_options
          OPTION 
            value: n
            style: {}
            n

    DIV null,

      INPUT 
        key: 'filter'
        type: 'text'
        defaultValue: "" 
        onChange: (e) => 
          @local.filter_channels = e.target.value
          save @local

    for channel_info in all

      if @local.filters[ channel_info.auto or ''  ]

        if (@local.mentioned_in == ' ' or @local.mentioned_in in channel_info.mentioned_in) and 
           (@local.included_in == ' ' or @local.included_in in channel_info.included_in)
          
          if !@local.filter_channels || (channel_info.title?.toLowerCase().indexOf(@local.filter_channels.toLowerCase()) > -1)

            CHANNEL
              key: channel_info.channelId
              channel_info: channel_info


dom.CHANNEL = ->

  channel_info = @props.channel_info
  if channel_info.auto == 'exclude'
    color = '#ddb2b5'
  else if channel_info.auto == 'include'
    color = '#b9ddb2'
  else if channel_info.auto == 'eligible'
    color = '#e5e3f2'
  else
    color = '#fafafa'

  key = "/channel/#{channel_info.title}"
  retrieve(key)

  nfObject = new Intl.NumberFormat('en-US')

  DIV
    style:
      margin: '8px 12px'
      padding: '8px 12px'
      backgroundColor: color

    H4
      style:
        fontSize: 18


      A
        style: 
          color: 'black'

        href: "https://youtube.com/#{channel_info.customUrl}"
        target: '_blank'

        channel_info.title


    DIV null,
      "#{nfObject.format(channel_info.viewCount)} total views"
    DIV null,
      "#{nfObject.format(channel_info.subscriberCount)} subscribers"


    DIV null,

      SPAN null,
        "Included in"
        UL null,
          for song in channel_info.included_in or []
            LI  
              style:
                listStyle: 'none'
                padding: '0 6px'
              song

      SPAN null,
        "Mentioned in"
        UL null,
          for song in channel_info.mentioned_in or []
            LI  
              style:
                listStyle: 'none'
                padding: '0 6px'
              song

    UL null,
      for note, title of channel_info.notes or []
        title ?= '<none>'
        if title
          LI  
            style:
              listStyle: 'none'
              padding: '0 6px'

            A 
              target: "_blank"
              href: "https://www.youtube.com/channel/#{channel_info.channelId}/search?query=#{note}"
              "#{title}"


    DIV null,

      SELECT
        defaultValue: channel_info.auto
        style: 
          fontSize: 18
          display: 'block'
        onChange: (e) => 
          channel_info.auto = e.target.value
          to_save = 
              key: key
              val: channel_info
          save to_save

        for val in ['', 'eligible', 'include', 'exclude']
          OPTION
            key: val
            value: val
            val 

