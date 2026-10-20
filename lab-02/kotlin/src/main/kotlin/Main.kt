import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.util.Locale
import java.util.PriorityQueue
import java.util.TreeMap
import javax.imageio.ImageIO
import kotlin.math.*

data class Expr(val kind: String, val text: String = "", val children: List<Expr> = emptyList())
data class Bound(val lo: Double, val hi: Double, val lip: Double)

class ExpressionParser(expression: String) {
    private val source = expression
    private var pos = 0
    private fun skip() { while (pos < source.length && source[pos].isWhitespace()) pos++ }
    private fun take(token: String): Boolean {
        skip()
        if (!source.startsWith(token, pos)) return false
        pos += token.length
        return true
    }
    fun parse(): Expr {
        val node = sum()
        skip()
        require(pos == source.length) { "Лишний символ в позиции $pos" }
        return node
    }
    private fun sum(): Expr {
        var node = product()
        while (true) node = when {
            take("+") -> Expr("binary", "+", listOf(node, product()))
            take("-") -> Expr("binary", "-", listOf(node, product()))
            else -> return node
        }
    }
    private fun product(): Expr {
        var node = unary()
        while (true) {
            skip()
            node = when {
                source.startsWith("**", pos) -> return node
                take("*") -> Expr("binary", "*", listOf(node, unary()))
                take("/") -> Expr("binary", "/", listOf(node, unary()))
                else -> return node
            }
        }
    }
    private fun unary(): Expr = when {
        take("+") -> Expr("unary", "+", listOf(unary()))
        take("-") -> Expr("unary", "-", listOf(unary()))
        else -> power()
    }
    private fun power(): Expr {
        val node = atom()
        return if (take("**")) Expr("binary", "**", listOf(node, unary())) else node
    }
    private fun atom(): Expr {
        skip()
        if (take("(")) {
            val node = sum()
            require(take(")")) { "Нет закрывающей скобки" }
            return node
        }
        require(pos < source.length) { "Неожиданный конец выражения" }
        if (source[pos].isDigit() || source[pos] == '.') {
            val pattern = Regex("(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
            val match = pattern.find(source, pos)
            require(match != null && match.range.first == pos) { "Неверное число" }
            pos = match.range.last + 1
            require(match.value.toDouble().isFinite()) { "Неконечная константа" }
            return Expr("number", match.value)
        }
        val start = pos
        while (pos < source.length && source[pos].isLetter()) pos++
        require(pos > start) { "Недопустимый символ в позиции $pos" }
        val name = source.substring(start, pos)
        if (name in setOf("x", "pi", "e")) return Expr("name", name)
        require(name in setOf("sin", "cos", "exp", "log", "sqrt", "abs") && take("(")) {
            "Неизвестное имя или вызов: $name"
        }
        val argument = sum()
        require(take(")")) { "Нет закрывающей скобки функции" }
        return Expr("call", name, listOf(argument))
    }
}

fun evaluate(node: Expr, x: Double): Double {
    val args = node.children.map { evaluate(it, x) }
    return when (node.kind) {
        "number" -> node.text.toDouble()
        "name" -> when (node.text) { "x" -> x; "pi" -> PI; else -> E }
        "unary" -> if (node.text == "-") -args[0] else args[0]
        "call" -> when (node.text) {
            "sin" -> sin(args[0]); "cos" -> cos(args[0]); "exp" -> exp(args[0])
            "log" -> ln(args[0]); "sqrt" -> sqrt(args[0]); else -> abs(args[0])
        }
        else -> when (node.text) {
            "+" -> args[0] + args[1]; "-" -> args[0] - args[1]
            "*" -> args[0] * args[1]; "/" -> args[0] / args[1]; else -> args[0].pow(args[1])
        }
    }
}

fun expressionBound(node: Expr, a: Double, b: Double): Bound {
    // Диапазон выражения и аналитическая верхняя оценка его константы Липшица.
    if (node.kind == "number") return Bound(node.text.toDouble(), node.text.toDouble(), 0.0)
    if (node.kind == "name") return when (node.text) {
        "x" -> Bound(a, b, 1.0); "pi" -> Bound(PI, PI, 0.0); else -> Bound(E, E, 0.0)
    }
    val u = expressionBound(node.children[0], a, b)
    if (node.kind == "unary") return if (node.text == "-") Bound(-u.hi, -u.lo, u.lip) else u
    if (node.kind == "call") return when (node.text) {
        "sin", "cos" -> Bound(-1.0, 1.0, u.lip)
        "exp" -> Bound(exp(u.lo), exp(u.hi), exp(u.hi) * u.lip)
        "log" -> { require(u.lo > 0) { "Для log задайте корректное L вручную" }; Bound(ln(u.lo), ln(u.hi), u.lip/u.lo) }
        "sqrt" -> { require(u.lo > 0) { "Для sqrt задайте корректное L вручную" }; Bound(sqrt(u.lo), sqrt(u.hi), u.lip/(2*sqrt(u.lo))) }
        else -> Bound(if (u.lo <= 0 && u.hi >= 0) 0.0 else min(abs(u.lo),abs(u.hi)), max(abs(u.lo),abs(u.hi)),u.lip)
    }
    val v = expressionBound(node.children[1], a, b)
    val m = max(abs(u.lo), abs(u.hi)); val n = max(abs(v.lo), abs(v.hi))
    return when (node.text) {
        "+" -> Bound(u.lo+v.lo, u.hi+v.hi, u.lip+v.lip)
        "-" -> Bound(u.lo-v.hi, u.hi-v.lo, u.lip+v.lip)
        "*" -> {
            val values = listOf(u.lo*v.lo,u.lo*v.hi,u.hi*v.lo,u.hi*v.hi)
            Bound(values.min(),values.max(),n*u.lip+m*v.lip)
        }
        "/" -> {
            require(v.lo > 0 || v.hi < 0) { "Знаменатель может быть нулевым; задайте L вручную" }
            val values = listOf(u.lo/v.lo,u.lo/v.hi,u.hi/v.lo,u.hi/v.hi)
            val d = min(abs(v.lo),abs(v.hi))
            Bound(values.min(),values.max(),u.lip/d+m*v.lip/(d*d))
        }
        else -> {
            require(v.lip == 0.0 && v.lo == v.hi && v.lo == floor(v.lo)) { "Для этой степени задайте L вручную" }
            val p = v.lo
            when {
                p == 0.0 -> Bound(1.0,1.0,0.0)
                p > 0 -> {
                    val values = mutableListOf(u.lo.pow(p),u.hi.pow(p))
                    if (u.lo <= 0 && u.hi >= 0) values.add(0.0)
                    Bound(values.min(),values.max(),p*m.pow(p-1)*u.lip)
                }
                else -> {
                    require(u.lo > 0 || u.hi < 0) { "Отрицательная степень при возможном нуле" }
                    val values = listOf(u.lo.pow(p),u.hi.pow(p))
                    Bound(values.min(),values.max(),abs(p)*min(abs(u.lo),abs(u.hi)).pow(p-1)*u.lip)
                }
            }
        }
    }
}

data class Interval(val lower: Double, val xl: Double, val xr: Double,
                    val fl: Double, val fr: Double, val candidate: Double)
data class Step(val iteration: Int, val upper: Double, val lower: Double,
                val xBest: Double, val candidate: Double)
data class Result(val expression: String, val a: Double, val b: Double, val eps: Double,
                  val lip: Double, val direction: String, val x: Double, val value: Double,
                  val lower: Double, val upper: Double, val gap: Double,
                  val iterations: Int, val evaluations: Int, val seconds: Double,
                  val samples: List<Pair<Double,Double>>, val history: List<Step>)

fun intersection(xl: Double, fl: Double, xr: Double, fr: Double, lip: Double): Interval {
    val x = (xl+xr)/2 - (fr-fl)/(2*lip)
    val lower = (fl+fr-lip*(xr-xl))/2
    return Interval(lower,xl,xr,fl,fr,x)
}

fun piyavskii(expression: String, a: Double, b: Double, eps: Double = 0.01,
              suppliedL: Double? = null, direction: String = "min", maxIter: Int = 100000): Result {
    require(a.isFinite() && b.isFinite() && a < b) { "Нужен конечный отрезок a < b" }
    require(eps.isFinite() && eps > 0 && maxIter > 0) { "eps и maxIter должны быть положительными" }
    require(direction in setOf("min", "max")) { "Направление: min или max" }
    val node = ExpressionParser(expression).parse()
    val lip = suppliedL ?: max(expressionBound(node,a,b).lip*(1+1e-12),1e-12)
    require(lip.isFinite() && lip > 0) { "L должно быть конечным и положительным" }
    val sign = if (direction == "min") 1.0 else -1.0
    fun value(x: Double): Double = (sign*evaluate(node,x)).also {
        require(it.isFinite()) { "Функция должна быть конечной на отрезке" }
    }
    val started = System.nanoTime()
    val fa = value(a); val fb = value(b)
    val samples = TreeMap<Double,Double>().apply { put(a,fa); put(b,fb) }
    var xBest = if (fa <= fb) a else b
    var upper = min(fa,fb)
    val queue = PriorityQueue(compareBy<Interval> { it.lower }.thenBy { it.xl }.thenBy { it.xr })
    val history = mutableListOf<Step>()
    fun addInterval(xl: Double, fl: Double, xr: Double, fr: Double) {
        require(abs(fr-fl) <= lip*(xr-xl)+1e-10*maxOf(1.0,abs(fl),abs(fr))) { "L противоречит известным пробам" }
        queue.add(intersection(xl,fl,xr,fr,lip))
    }
    addInterval(a,fa,b,fb)
    var iteration = 0
    while (true) {
        val current = queue.peek()
        history.add(Step(iteration,upper,current.lower,xBest,current.candidate))
        // Остановка по разрыву доказанных оценок, а не по расстоянию между пробами.
        if (upper-current.lower <= eps) break
        check(iteration < maxIter) { "Превышен maxIter; точность не достигнута" }
        queue.remove()
        val x = current.candidate
        check(x > current.xl && x < current.xr && !samples.containsKey(x)) { "Предел машинной точности" }
        val fx = value(x)
        samples[x] = fx
        if (fx < upper) { xBest = x; upper = fx }
        // Выбранный интервал заменяется двумя; остальные вершины не пересчитываются.
        addInterval(current.xl,current.fl,x,fx)
        addInterval(x,fx,current.xr,current.fr)
        iteration++
    }
    val elapsed = (System.nanoTime()-started)/1e9
    val lower = queue.peek().lower
    return Result(expression,a,b,eps,lip,direction,xBest,sign*upper,
        if (sign == 1.0) lower else -upper, if (sign == 1.0) upper else -lower,
        upper-lower,iteration,samples.size,elapsed,samples.map { it.key to sign*it.value },history)
}

fun plotResult(result: Result, path: String) {
    // Сетка нужна только для рисунка: оптимизация выше её не использует.
    val node = ExpressionParser(result.expression).parse()
    val sign = if (result.direction == "min") 1.0 else -1.0
    val polygon = mutableListOf(result.samples.first())
    result.samples.zipWithNext().forEach { (left,right) ->
        val v = intersection(left.first,sign*left.second,right.first,sign*right.second,result.lip)
        polygon.add(v.candidate to sign*v.lower); polygon.add(right)
    }
    val curve = (0..2500).map { i ->
        val x = result.a+(result.b-result.a)*i/2500.0
        x to evaluate(node,x)
    }
    val image = BufferedImage(1600,650,BufferedImage.TYPE_INT_RGB)
    val g = image.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON)
    g.color = Color.WHITE; g.fillRect(0,0,image.width,image.height)
    fun panel(left: Int, xmin: Double, xmax: Double, ymin: Double, ymax: Double, title: String) {
        val top = 70; val width = 650; val height = 440
        fun px(x: Double) = left+((x-xmin)/(xmax-xmin)*width).toInt()
        fun py(y: Double) = top+height-((y-ymin)/(ymax-ymin)*height).toInt()
        g.font = Font("SansSerif",Font.PLAIN,18)
        for (i in 0..5) {
            val x = xmin+(xmax-xmin)*i/5; val y = ymin+(ymax-ymin)*i/5
            g.color = Color(225,230,235); g.drawLine(px(x),top,px(x),top+height);g.drawLine(left,py(y),left+width,py(y))
            g.color = Color.DARK_GRAY
            g.drawString("%.2f".format(Locale.US,x),px(x)-23,top+height+30)
            g.drawString("%.2f".format(Locale.US,y),left-65,py(y)+5)
        }
        g.color = Color.DARK_GRAY; g.drawRect(left,top,width,height)
        g.font = Font("SansSerif",Font.BOLD,21);g.drawString(title,left,38)
        val oldClip = g.clip
        g.clipRect(left,top,width,height)
        fun line(points: List<Pair<Double,Double>>, color: Color, thickness: Float) {
            g.color = color; g.stroke = BasicStroke(thickness)
            points.zipWithNext().forEach { (u,v) -> g.drawLine(px(u.first),py(u.second),px(v.first),py(v.second)) }
        }
        line(curve,Color(29,79,145),2.5f);line(polygon,Color(212,122,20),1.4f)
        g.color = Color(120,135,150)
        result.samples.forEach { (x,y) -> g.fillOval(px(x)-2,py(y)-2,4,4) }
        g.color = Color(187,34,51);g.fillOval(px(result.x)-7,py(result.value)-7,14,14)
        g.clip = oldClip
        g.font = Font("SansSerif",Font.PLAIN,18)
        g.color = Color(29,79,145);g.drawString("Синий: f(x)",left,610)
        g.color = Color(212,122,20);g.drawString("Оранжевый: ломаная",left+155,610)
        g.color = Color(187,34,51);g.drawString("Красный: экстремум",left+390,610)
        g.color = Color.DARK_GRAY;g.drawString("x",left+width/2,top+height+58);g.drawString("f(x)",left-65,top-15)
    }
    val ys = (curve+polygon).map { it.second }
    val padding = max(1.0,(ys.max()-ys.min())*0.05)
    panel(85,result.a,result.b,ys.min()-padding,ys.max()+padding,"Kotlin: весь отрезок")
    val zoom = min((result.b-result.a)/4,0.15)
    panel(885,max(result.a,result.x-zoom),min(result.b,result.x+zoom),
          result.value-max(0.03,3*result.eps),result.value+max(0.1,10*result.eps),"Увеличение около экстремума")
    g.dispose()
    File(path).absoluteFile.parentFile.mkdirs()
    ImageIO.write(image,"png",File(path))
}

fun selfTest() {
    data class Case(val f: String,val a: Double,val b: Double,val mode: String,val exact: Double)
    val cases = listOf(Case("(x-0.37)**2",-1.0,2.0,"min",0.0),Case("x",-2.0,3.0,"min",-2.0),
        Case("5",-1.0,1.0,"min",5.0),Case("-(x-0.2)**2",-1.0,1.0,"max",0.0),Case("abs(x-0.3)",-1.0,1.0,"min",0.0))
    for (c in cases) {
        val r = piyavskii(c.f,c.a,c.b,1e-4,direction=c.mode)
        check(r.lower <= c.exact+1e-9 && c.exact <= r.upper+1e-9 && abs(r.value-c.exact)<=1e-4 && r.gap<=1e-4)
        println("OK: ${c.f}, ${c.mode}, iterations=${r.iterations}")
    }
    check(evaluate(ExpressionParser("-x**2").parse(),2.0) == -4.0)
    check(evaluate(ExpressionParser("2**3**2").parse(),0.0) == 512.0)
}

fun main(args: Array<String>) {
    Locale.setDefault(Locale.US)
    if (args.firstOrNull() == "--self-test") { selfTest(); return }
    require(args.isEmpty() || args.size in 4..7) { "Аргументы: expression a b eps [min|max] [L|auto] [png]" }
    val expression = args.getOrNull(0) ?: "x**2 + 10*(1-cos(2*pi*x))"
    val a = args.getOrNull(1)?.toDouble() ?: -2.0
    val b = args.getOrNull(2)?.toDouble() ?: 2.0
    val eps = args.getOrNull(3)?.toDouble() ?: 0.01
    val mode = args.getOrNull(4) ?: "min"
    val suppliedL = args.getOrNull(5)?.takeUnless { it == "auto" }?.toDouble()
    val path = args.getOrNull(6) ?: "kotlin-visualization.png"
    val r = piyavskii(expression,a,b,eps,suppliedL,mode)
    println("expression=$expression")
    println("x=${r.x}\nf=${r.value}\nL=${r.lip}\nlower=${r.lower}\nupper=${r.upper}\ngap=${r.gap}")
    println("iterations=${r.iterations}\nevaluations=${r.evaluations}\nseconds=${r.seconds}")
    plotResult(r,path)
    println("plot=$path")
}
